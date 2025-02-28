import os
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.nn.modules.loss import CrossEntropyLoss
import cv2
import numpy as np
from utils.loss import Dice_loss

from tensorboardX import SummaryWriter
from utils.metrics import *
from models.HCLHRL import HCLHRL_unet
import json
import logging
from utils.plot import *
import argparse
from utils.dataset import *
import math
import datetime


os.chdir(sys.path[0])
# from torchstat import stat

def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument("--action", type=str, help="train/test/train&test", default="train")
    parse.add_argument("--backbone", type=str, help="unet/densenet/cenet", default="auto")
    parse.add_argument("--arch", type=str, help="models:A_unet/B_unet/C_unet/D_unet/E_unet/F_unet/G_unet", default="icassp_unet_cls")
    parse.add_argument("--epochs", type=int, default=100)
    parse.add_argument("--batch_size", type=int, default=4)
    parse.add_argument("--lr", type=float, default=0.001)
    parse.add_argument("--lama", type=float, default=1.0)
    parse.add_argument("--gama", type=float, default=0.01)
    parse.add_argument("--patchsize", type=str, default='3')
    parse.add_argument("--save_epoch", type=int, default=100)
    parse.add_argument("--predict_output_path", type=str, default='/data16t/usr/HCLHRL/predict')
    parse.add_argument("--plot_save_path", type=str, default='/data16t/usr/HCLHRL/plot')
    parse.add_argument("--weight_save_path", type=str, default='/data16t/usr/HCLHRL/weights')
    parse.add_argument("--writer_dir", type=str, default='/data16t/usr/HCLHRL/runs')
    parse.add_argument('--dataset', default='tn3k',
                       help='dataset name:DDTI/tn3k/ZY')
    parse.add_argument('--DDTIfile', default='/data16t/zelan/DisSimNet/data/DDTI/DDTI.json')
    parse.add_argument('--seed', type=int,  default=2022, help='random seed')
    parse.add_argument("--log_dir", default='../log', help="log dir")
    parse.add_argument("--use_pretrain", default=False, help="use pretrain model")
    parse.add_argument("--pretrain", default='kp_min_loss.pth', help="detect pretrain model path")
    parse.add_argument("--threshold", type=float, default=0.5)
    args = parse.parse_args()
    return args

def log_args(args, txt_path):
    with open(txt_path, 'a') as f:
        f.write("Experiment arguments:\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
        f.write("\n" + "-"*50 + "\n")

def getLog(args):
    dirname = os.path.join(args.log_dir,args.arch, str(args.dataset))
    filename = dirname +'log.log'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    logging.basicConfig(
            filename=filename,
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
    return logging

def split_dataset(dataset, val_split=0.2, test_split=0.1):
    dataset_len = len(dataset)
    val_len = int(dataset_len * val_split)
    test_len = int(dataset_len * test_split)
    train_len = dataset_len - val_len - test_len
    return random_split(dataset, [train_len, val_len, test_len])

def load_split_indices(filename):
    with open(filename, 'r') as f:
        indices = json.load(f)
    return indices

def create_dataloaders(dataset, indices, batch_size, num_workers=4):
    train_indices = indices['train']
    val_indices = indices['val']
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=1, num_workers=num_workers, shuffle=False)
    return train_loader, val_loader


def get_models(args):
    if args.backbone == 'unet':
        model = HCLHRL_unet(3,1).to(device)
    return model


def calculate_metrics(predict, gt):
    """
    Calculate various metrics including mIoU, Dice, Precision, and Hausdorff Distance.

    Args:
    predict (torch.Tensor): Prediction tensor, binary (0 or 1) on GPU.
    gt (torch.Tensor): Ground truth tensor, binary (0 or 1) on GPU.

    Returns:
    dict: A dictionary containing mIoU, Dice, Precision, and Hausdorff Distance.
    """
    # Calculate intersection and union for IoU
    intersection = torch.logical_and(predict, gt).sum().float()
    union = torch.logical_or(predict, gt).sum().float()
    miou = (intersection / union) if union != 0 else torch.tensor(0.0, device=predict.device)

    # Calculate Dice coefficient
    dice = (2. * intersection / (predict.sum() + gt.sum())) if (predict.sum() + gt.sum()) != 0 else torch.tensor(0.0, device=predict.device)

    # Calculate Precision
    precision = (intersection / predict.sum()) if predict.sum() != 0 else torch.tensor(0.0, device=predict.device)

    # Hausdorff Distance - using CPU fallback since no native GPU implementation in PyTorch
    hd1 = directed_hausdorff(predict.cpu().numpy(), gt.cpu().numpy())[0]
    hd2 = directed_hausdorff(gt.cpu().numpy(), predict.cpu().numpy())[0]
    hausdorff_distance = max(hd1, hd2)
    hd = hausdorff_distance(predict, gt, distance='euclidean')

    # Gather all metrics in a dictionary
    metrics = {
        'mIoU': miou.item(),  # Convert to Python float for easier handling outside Torch
        'Dice': dice.item(),
        'Precision': precision.item(),
        'Hausdorff Distance': hd
    }

    return metrics

def save_correlation_image(correlation_map, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)
    # 将相关性矩阵转换为numpy数组
    correlation_map_np = correlation_map.detach().cpu().numpy()
    # 取第一个batch的图像
        # 检查correlation_map的维度
    if len(correlation_map_np.shape) == 3:
        b, h ,w = correlation_map.size()
        for idx, corr in enumerate(correlation_map):
            img = correlation_map_np[idx]  # 假设我们只取相关性矩阵的第一个通道
            # 归一化到[0, 1]区间
            img = (img - img.min()) / (img.max() - img.min())
            # 保存图像
            plt.imsave(os.path.join(path, os.path.basename(filename[idx])), img, cmap='hot')  # 使用热图颜
    elif len(correlation_map_np.shape) == 2:
        # 不带batch的情况，直接处理
        img = correlation_map_np
        # 归一化到[0, 1]区间
        img = (img - img.min()) / (img.max() - img.min())
        # 保存图像
        plt.imsave(os.path.join(path, os.path.basename(filename[0])), img, cmap='hot')
    else:
        raise ValueError("correlation_map should be 2D or 3D.")    
        
def save_correlation_image_(correlation_map, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)
    # 将相关性矩阵转换为numpy数组
    correlation_map_np = correlation_map.detach().cpu().numpy()
    # 取第一个batch的图像
    if len(correlation_map_np.shape) == 3:
        b, h ,w = correlation_map.size()
        for idx, corr in enumerate(correlation_map):
            img = correlation_map_np[idx]  # 假设我们只取相关性矩阵的第一个通道
            # 归一化到[0, 1]区间
            img = (img - img.min()) / (img.max() - img.min())
            img = 1-img
            # 保存图像
            plt.imsave(os.path.join(path, os.path.basename(filename[idx])), img, cmap='hot')  # 使用热图颜
    elif len(correlation_map_np.shape) == 2:
        # 不带batch的情况，直接处理
        img = correlation_map_np
        # 归一化到[0, 1]区间
        img = (img - img.min()) / (img.max() - img.min())
        img = 1-img
        # 保存图像
        plt.imsave(os.path.join(path, os.path.basename(filename[0])), img, cmap='hot')
    else:
        raise ValueError("correlation_map should be 2D or 3D.")

def train(args, model, train_data_loader, val_dataloader, optimizer, writer, device):
    total_loss = 0
    min_avg_loss = float('inf')
    min_avg_loss_seg = float('inf')
    loss_list = []
    iou_list = []
    dice_list = []
    hd_list = []
    precision_list = []
    best_iou = 0
    model.train()
    loss_S = torch.nn.BCELoss()
    ce_loss = CrossEntropyLoss(ignore_index=2)
    dice_loss = pDLoss(n_classes=2, ignore_index=2)

    for epoch in range(args.epochs):
        total_loss_seg = 0
        total_loss = 0
        min_loss = float('inf')
        max_loss = float('-inf')

        for index, (x, box, fore, back, gt, masks_path) in enumerate(train_data_loader):
            img, box, fore, back, gt = map(
                lambda t: t.to(device, non_blocking=True, dtype=torch.float32), 
                (x, box, fore, back, gt)
            )
            x_label = box.max(dim=2, keepdim=True)[0]
            y_label = box.max(dim=3, keepdim=True)[0]

            seg, out_x, out_y, loss_contrast, corr_fore_map, corr_back_map = model(img, fore, back)
            save_correlation_image(corr_fore_map,os.path.join(args.corr_path, str(args.arch)+str(args.name),'fore_correlation'), masks_path)
            save_correlation_image(corr_back_map,os.path.join(args.corr_path,  str(args.arch)+str(args.name), 'back_correlation'), masks_path)
            save_correlation_image_(corr_back_map,os.path.join(args.corr_path,  str(args.arch)+str(args.name), 'back_correlation_'), masks_path)

            loss_pair = loss_S(seg*fore, fore)
            print('loss_pair:%f' % (loss_pair))
            print('loss_contrast:%f' % (loss_contrast))
            corr_fore_map = torch.sigmoid(corr_fore_map)
            corr_back_map = torch.sigmoid(corr_back_map)
            corr_mask_f = corr_fore_map.unsqueeze(1)
            corr_mask_b = 1- corr_back_map.unsqueeze(1)
            corr_mask = corr_mask_b/2 + corr_mask_f/2
            
            loss_cls = loss_S(corr_mask, seg)
            loss_seg = Dice_loss(out_x,x_label) + Dice_loss(out_y,y_label)
            loss_seg = loss_seg.mean()
            print('loss_alignment:%f' % (loss_seg))
            print('loss_cls:%f' % (loss_cls))

            loss = loss_seg + loss_pair + args.lama * loss_contrast + args.gama * loss_cls

            print('loss_iteration:',loss)
            print('-------------------')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        loss_list.append(total_loss)
        writer.add_scalar('epoch_loss', total_loss, global_step=epoch)
        avg_loss_seg = total_loss_seg / len(train_data_loader)
        avg_loss = total_loss / len(train_data_loader)
        min_avg_loss_seg = min(min_avg_loss_seg, avg_loss_seg)
        min_avg_loss = min(min_avg_loss, avg_loss)
           
        best_iou, aver_iou, aver_dice, aver_hd, aver_precision = val(args, model, epoch, optimizer, loss, best_iou, val_dataloader)
        writer.add_scalar('aver_iou', aver_iou, global_step=epoch)
        writer.add_scalar('aver_dice', aver_dice, global_step=epoch)
        writer.add_scalar('aver_hd', aver_hd, global_step=epoch)
        writer.add_scalar('aver_precision', aver_precision, global_step=epoch)

        iou_list.append(aver_iou)
        dice_list.append(aver_dice)
        hd_list.append(aver_hd)
        precision_list.append(aver_precision)

        print("epoch %d loss:%0.3f" % (epoch, total_loss))
        logging.info("epoch %d loss:%0.3f" % (epoch, total_loss))
        print(
            'Epoch %d, photo number %d,avg loss %f, min loss %f, max loss %f,min avg loss %f' % (
            epoch, index + 1, avg_loss, min_loss, max_loss, min_avg_loss))
        print('-------------------')

    plot_save_path = os.path.join(str(args.plot_save_path),'HCLHRL' + '_' + str(args.backbone), str(args.dataset) + '_' + str(args.lama) + '_' + str(args.gama))
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)

    loss_plot(args.epochs, loss_list,plot_save_path)
    metrics_plot(plot_save_path, args.epochs, 'Miou&Dice&Precision', iou_list, dice_list, precision_list)
    metrics_plot(plot_save_path, args.epochs, 'HD', hd_list)


def val(args,model,epoch,optimizer,loss,best_iou,val_dataloaders):
    model = model.eval()
    with torch.no_grad():
        miou_total = 0
        hd_total = 0
        dice_total = 0
        precision_total = 0
 
        weight_save_path = args.weight_save_path + args.mode +args.patchsize
        if not os.path.exists(weight_save_path):
            os.makedirs(weight_save_path)
        num = len(val_dataloaders)
      
        for index, (x, box, fore, back, gt, masks_path) in enumerate(val_data_loader):

            img, box, fore, back, gt = map(
                lambda t: t.to(device, non_blocking=True, dtype=torch.float32), 
                (x, box, fore, back, gt)
            )

            seg, out_x, out_y, loss_contrast, corr_fore_map, corr_back_map = model(img, fore, back)
            masks = masks_path
            img_y = seg.cpu().detach().numpy()  #[4,256,256]
            img_y = img_y.squeeze()
            img_y[img_y < args.threshold] = 0
            predict_output_path = os.path.join(str(args.predict_output_path), 'HCLHRL' + '_' + str(args.backbone), str(args.dataset) + '_' + str(args.lama) + '_' + str(args.gama))
            if not os.path.exists(predict_output_path):
                os.makedirs(predict_output_path)
            plt.imsave(predict_output_path + '/' + masks[0].split('/')[-1], img_y, cmap='Greys_r')

            predictions_binary = (seg >= args.threshold).float()
            gt = (gt >= args.threshold).float()
            metrics = calculate_metrics(predictions_binary, gt)
            tem_hd = metrics['Hausdorff Distance']
            tem_iou = metrics['mIoU']
            tem_dice =metrics['Dice']
            tem_precision = metrics['Precision']
            print(str(args.arch)+'_'+str(args.dataset)+'_'+str(args.mode))
            print('tem_hd=%f,temp_iou=%f,tem_dice=%f,tem_precision=%f' % (tem_hd,tem_iou,tem_dice,tem_precision))

            hd_total += tem_hd
            miou_total += tem_iou
            dice_total += tem_dice
            precision_total += tem_precision
            aver_iou = miou_total / num
            aver_hd = hd_total / num
            aver_dice = dice_total/num
            aver_precision = precision_total/num

        print('##################')
        print('# ---- Mean ---- #')
        print('##################')

        print(str(args.arch)+'_'+str(args.dataset)+'_'+str(args.mode))
        print('Miou=%f,aver_hd=%f,aver_dice=%f,aver_precision=%f' % (aver_iou,aver_hd,aver_dice,aver_precision))
        logging.info('Miou=%f,aver_hd=%f,aver_dice=%f,aver_precision=%f' % (aver_iou,aver_hd,aver_dice,aver_precision))
        with open(txt_path, 'a') as f:
            f.write(f'Miou={aver_iou},aver_hd={aver_hd},aver_dice={aver_dice},aver_precision={aver_precision}\n')
        
        if aver_iou > best_iou :
            print('aver_iou:{} > best_iou:{}'.format(aver_iou,best_iou))
            best_iou = aver_iou
            logging.info('aver_iou:{} > best_iou:{}'.format(aver_iou,best_iou))
            logging.info('===========>save best model!')

            print('===========>save best model!')
            #torch.save(model.state_dict(), r'./saved_model/'+str(args.arch)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.epoch)+'.pth')
            torch.save({'epoch':epoch,
                        'model_state_dict': model.state_dict(),
                        'loss':loss,
                        'optimizer_state_dict':optimizer.state_dict()}, os.path.join(weight_save_path, str(args.arch) + '_' + str(args.dataset)+'_'+str(args.lama)+'_'+str(args.gama)+'.pth'))
        return best_iou,aver_iou,aver_dice,aver_hd,aver_precision
    

if __name__ == "__main__":
    f = torch.cuda.is_available()
    device = torch.device("cuda" if f else "cpu")
    args = getArgs()
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    arch = f'{args.arch}_{args.dataset}'
    txt_path = os.path.join(args.log_dir, 'ablation', arch, f'log_{current_time}.txt')
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    log_args(args, txt_path)
    writer_dir = os.path.join(str(args.writer_dir+str(args.mode)),str(args.arch),str(args.dataset))
    if not os.path.exists(writer_dir ):
        os.makedirs(writer_dir )
    writer = SummaryWriter(writer_dir)
    model = get_models(args)

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=0.00001)

    if args.dataset == 'tn3k':
        dataset = Dataset_all(os.path.join('..', 'data', str(args.dataset), 'train'))
        train_data_loader = DataLoader(dataset=dataset,
                                                        batch_size=args.batch_size,
                                                        num_workers=4,
                                                        shuffle=True)
        
        val_dataset = Dataset_all(os.path.join('..', 'data', str(args.dataset), 'test'))
        val_data_loader = DataLoader(dataset=val_dataset,
                                                        batch_size= 1,
                                                        shuffle=True,
                                                        num_workers=4)

    elif args.dataset in {'DDTI'}:
        dataset = Dataset_all(os.path.join('..', 'data', str(args.dataset)))
        # 加载索引
        indices = load_split_indices(args.DDTIfile)
        # 创建 DataLoader
        train_data_loader, val_data_loader = create_dataloaders(dataset, indices, args.batch_size)

    # 训练
    if args.action == "train":
        train(args,model,train_data_loader,val_data_loader,optimizer,writer,device)



