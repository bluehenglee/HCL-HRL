import torch
import os
import numpy as np
from torch import nn
import torchvision
import json
import torch.utils.data
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from segment_anything import sam_model_registry
import cv2
from utils_data import *
import argparse
from practical_function import unfold_wo_center,get_images_color_similarity,compute_pairwise_term

def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument("--data_root", 
                       type=str, 
                       help="dataset for sam, DDTI/ZY/TRFE/BUSI", 
                       default='/data16t/zelan/DisSimNet/data')
    parse.add_argument("--data_name", 
                       type=str, 
                       help="dataset root pach", 
                       default='DDTI')
    parse.add_argument("--save_path", 
                       type=str, 
                       help="sam results savepath", 
                       default='SAM_result')
    parse.add_argument("--prompt", type=str, help='box/point', default='box')
    parse.add_argument("--device", type=str, default="cuda:0", help="device")
    parse.add_argument("-chk",
                        "--checkpoint",
                        type=str,
                        default="work_dir/MedSAM/medsam_vit_b.pth",
                        help="path to the trained model")
    args = parse.parse_args()
    return args

def one_json_to_numpy(dataset_path):
    with open(dataset_path) as fp:
        json_data = json.load(fp)
        points = json_data['shapes']

    landmarks = []
    for point in points:
        for p in point['points'][0]:
            landmarks.append(p)

    # print(landmarks)
    landmarks = np.array(landmarks)
    return landmarks

def json_to_numpy(dataset_path):
    with open(dataset_path, 'r') as fp:
        json_data = json.load(fp)
        shapes = json_data['shapes']

    # 初始化一个列表来存储所有关键点组
    keypoint_groups = []

    # 遍历shapes列表，每四个点构成一个组
    for i in range(0, len(shapes), 4):
        # 检查是否有足够的点来构成一个关键点组
        if i + 3 < len(shapes):
            # 提取四个点
            points = [shape['points'] for shape in shapes[i:i+4]]
            # 将每个点的坐标从列表转换为浮点数
            points = [[float(coord) for coord in point] for sublist in points for point in sublist]
            # 将点的列表转换为NumPy数组，并添加到关键点组列表中
            keypoint_group = np.array(points)
            keypoint_groups.append(keypoint_group)

    # 将所有关键点组合并为一个NumPy数组
    keypoint_groups_array = np.array(keypoint_groups)

    return keypoint_groups_array

def points_to_box(keypoints):
    """
    通过4个关键点真值得到一个不旋转的矩形框。
    Args:
        keypoints: 关键点信息,包含4个点的坐标,形状为(4, 2)
    Returns:
        box: 表示前景矩形框
    """
    boxes = []
    for points in keypoints:
        # 获取所有点的x和y坐标
        x_coords = points[:, 0]
        y_coords = points[:, 1]

        # 计算最小和最大值
        x_min = int(min(x_coords))
        x_max = int(max(x_coords))
        y_min = int(min(y_coords))
        y_max = int(max(y_coords))

        box = [x_min,y_min,x_max,y_max]
        boxes.append(box)
    return boxes

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    # 将掩膜合并（如需要），例如通过将所有掩膜相加
    if len(medsam_seg.shape) == 3:
        medsam_seg = np.sum(medsam_seg, axis=0)  # 将多个掩膜合并为一个，形状为 (B, H, W)
    return medsam_seg

@torch.no_grad()
def medsam_inference_noprompt(medsam_model, img_embed, H, W):
    # 使用 medsam 模型的解码器直接进行推理，不使用提示框
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=None,  # 不使用稀疏提示
        dense_prompt_embeddings=None,  # 不使用稠密提示
        multimask_output=False,
    )

    # 对预测的 logits 进行 sigmoid 转换，得到分割结果的概率图
    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    # 将预测的低分辨率掩膜上采样到原图尺寸
    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, H, W)
    
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (H, W)
    
    # 二值化处理，将概率值大于 0.5 的区域视为前景，生成最终的分割掩膜
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)

    return medsam_seg

def create_folder(args):
    if args.data_name == 'tn3k':
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'train','box')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'train','box'))
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'train','gt')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'train','gt'))
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'train','fore')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'train','fore'))
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'train','back')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'train','back'))
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'train','SAM_result')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'train','SAM_result'))
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'test','box')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'test','box'))
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'test','gt')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'test','gt'))
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'test','fore')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'test','fore'))
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'test','back')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'test','back'))
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'test','SAM_result')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'test','SAM_result'))
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'test','SAM_box')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'test','SAM_box'))
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'test','SAM_fore')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'test','SAM_fore'))
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'test','SAM_back')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'test','SAM_back'))
    elif args.data_name == 'DDTI':
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'box')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'box'))
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'gt')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'gt'))
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'fore')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'fore'))
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'back')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'back'))
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'SAM_result')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'SAM_result'))
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'SAM_box')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'SAM_box'))
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'SAM_fore')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'SAM_fore'))
        if not os.path.exists(os.path.join('.', 'data', str(args.data_name),'SAM_back')):
            os.makedirs(os.path.join('.', 'data', str(args.data_name),'SAM_back'))
class Dataset_processing(torch.utils.data.Dataset):
    # 初始化
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.img_name_list = sorted(os.listdir(os.path.join(dataset_path, 'imgs')))
        print(self.img_name_list)
    # 根据 index 返回位置的图像和label
    def __getitem__(self, index):
        # 先处理img
        img_path = os.path.join(self.dataset_path, 'imgs', self.img_name_list[index])
        img = cv2.imread(img_path)
        width, height = img.shape[1], img.shape[0]#h,w,c高度（rows）y,宽度（columns)x,通道数（channels)c
        img = cv2.resize(img, (256, 256))
        if len(img.shape) == 2:
            img_3c = np.repeat(img[:, :, None], 3, axis=-1)
        else:
            img_3c = img

        image = cv2.imread(img_path,0)#h,w 高度（rows）y,宽度（columns)x
        image = cv2.resize(image, (256, 256))
        image = image / 255
        # 读入标签
        label_path = os.path.join(self.dataset_path, 'labels', self.img_name_list[index].split('.')[0]+'.json')
        img_name = self.img_name_list[index]
        masks_path = os.path.join(self.dataset_path, 'masks', img_name)
        gt_path = os.path.join(self.dataset_path, 'gt', img_name)
        gt = cv2.imread(masks_path,0)
        gt = cv2.resize(gt, (256, 256))
        plt.imsave(gt_path, gt, cmap='Greys_r')
        box_mask = os.path.join(self.dataset_path, 'box', img_name)
        fore_mask = os.path.join(self.dataset_path, 'fore', img_name)
        back_mask = os.path.join(self.dataset_path, 'back', img_name)
        sam_mask = os.path.join(self.dataset_path, 'SAM_result', img_name)
        sam_box = os.path.join(self.dataset_path, 'SAM_box', img_name)
        sam_fore = os.path.join(self.dataset_path, 'SAM_fore', img_name)
        sam_back = os.path.join(self.dataset_path, 'SAM_back', img_name)

        point_groups = json_to_numpy(label_path)
        # 计算resize前后图片尺寸的比例
        new_width, new_height = 256, 256  # 调整后的图片尺寸，这里以256*256为例
        width_scale = new_width / width
        height_scale = new_height / height

        # 根据比例调整标签中关键点的坐标
        point_groups[:, :, 0] *= width_scale  # Scale x coordinates
        point_groups[:, :, 1] *= height_scale # Scale y coordinates

        fore = point_to_fore(point_groups,fore_mask)
        back = points_to_back(point_groups,back_mask)
        box = points_to_box(point_groups,box_mask)
        

        point_groups = torch.tensor(point_groups, dtype=torch.float32)
        # 转换为NumPy数组
        box = points_to_box(point_groups)
        box = np.array(box) #(1,4)

        print(point_groups.shape)
        # 初始化一个空数组来存储所有关键点
        all_points = []
        for points in point_groups:
            points = points.reshape(-1, 2)
            all_points.extend(points)
        # 转换为NumPy数组
        all_points = np.array(all_points)

        print(all_points.shape)
        if img_path.split('.')[0] != label_path.split('.')[0]:
            print("数据不一致")

        img_256 = transform.resize(
            img_3c, (256, 256), order=3, preserve_range=True, anti_aliasing=True
            ).astype(np.uint8)
        img_256 = (img_256 - img_256.min()) / np.clip(
            img_256.max() - img_256.min(), a_min=1e-8, a_max=None
            ) 
        # convert the shape to (3, H, W)
        img_1024_tensor = (torch.tensor(img_256).float().permute(2, 0, 1).unsqueeze(0).to(device))
        with torch.no_grad():
            image_embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)
            if args.prompt == 'box':
                medsam_seg = medsam_inference(medsam_model, image_embedding, box, height, width)
            elif args.prompt == 'none':
                medsam_seg = medsam_inference_noprompt(medsam_model, image_embedding, height, width)

        if not os.path.exists(os.path.join(self.dataset_path, args.save_path)):
            os.makedirs(os.path.join(self.dataset_path, args.save_path))

        plt.imsave(sam_mask, medsam_seg, cmap='Greys_r')#黑白

        result_box = np.zeros(fore.shape, dtype=np.uint8)
        result_fore = np.zeros(fore.shape, dtype=np.uint8)
        result_back = np.zeros(fore.shape, dtype=np.uint8)


        box_nonzero = box >= 127
        fore_nonzero = fore >= 127
        back_nonzero = back >= 127
        sam_nonzero = medsam_seg >= 127
        combined_box = np.logical_and(box_nonzero, sam_nonzero)
        combined_fore = np.logical_and(fore_nonzero, sam_nonzero)
        combined_back = np.logical_and(back_nonzero, np.logical_not(sam_nonzero))
        result_box[combined_box] = 255
        result_fore[combined_fore] = 255
        result_back[combined_back] = 255

        # save results output_folder
        cv2.imwrite(sam_box, result_box)
        cv2.imwrite(sam_fore, result_fore)
        cv2.imwrite(sam_back, result_back)

        print(f"Processed: {img_name}")
        # return img, masks_path
        return img, result_box, result_fore, result_back, masks_path
    
    def __len__(self):
        return len(self.img_name_list)

if __name__ =="__main__":
    f = torch.cuda.is_available()
    device = torch.device("cuda" if f else "cpu")
    args = getArgs()
    create_folder(args)
    #train and test
    medsam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    medsam_model = medsam_model.to(device)
    medsam_model.eval()

    if args.data_name == 'tn3k':
        train_dataset = Dataset_processing(os.path.join('.', 'data', str(args.data_name), 'train'))#tn3k
        val_dataset = Dataset_processing(os.path.join('.', 'data', str(args.data_name), 'test'))#tn3k
        train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=8,
                                                        shuffle=True)
        val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                        batch_size=8,
                                                        shuffle=True)
        for index,(img, masks_path) in enumerate(train_data_loader):
            print('preprocessing path data:'%(masks_path))
            
        for index,(img, masks_path) in enumerate(val_data_loader):
            print('preprocessing valid data:'%(masks_path))


    elif args.data_name == 'DDTI':
        dataset = Dataset_processing(os.path.join('.', 'data', str(args.data_name)))
        all_data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                        batch_size=8,
                                                        shuffle=True)
        for index,(img, masks_path) in enumerate(all_data_loader):
            print('preprocessing:'%(masks_path))
