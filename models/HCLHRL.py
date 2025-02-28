import torch
import os
from abc import ABC
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import models
import torch.nn.functional as F
from utils.practical_function import *
from functools import partial
f = torch.cuda.is_available()
from einops import rearrange, repeat, reduce
import numpy as np
from collections import OrderedDict

device = torch.device("cuda" if f else "cpu")

# import Constants
nonlinearity = partial(F.relu, inplace=True)


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.max_pool_conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out

class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity
        #棋盘格效应通常是由于转置卷积层（nn.ConvTranspose2d）的特定配置引起的，尤其是当使用奇数大小的卷积核和较大的步长时。
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        # self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 4, stride=2, padding=1)
        # self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, stride=2)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class MLP(nn.Module):
    def __init__(self, dim, projection_size=128, hidden_size=4096):
        super(MLP, self).__init__()
        
        # 定义网络结构
        self.network = nn.Sequential(
            nn.Linear(dim, hidden_size),       # 输入层到隐藏层
            nn.BatchNorm1d(hidden_size),       # 批量归一化
            nn.ReLU(inplace=True),             # 激活函数
            nn.Linear(hidden_size, projection_size)  # 隐藏层到输出层
        )

    def forward(self, x):
        return self.network(x)

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, drop_rate):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.drop_rate = drop_rate
    
    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        if self.drop_rate > 0:
            out = nn.Dropout2d(self.drop_rate)(out)
        return torch.cat([x, out], 1)
    
class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate, drop_rate))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        return self.pool(self.conv(self.relu(self.bn(x))))

class projection_head(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=2048, out_dim=2048):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, input):
        if torch.is_tensor(input):
            x = input
        else:
            x = input[-1]
            b = x.size()[0]
            x = F.adaptive_avg_pool3d(x, (1, 1, 1)).view(b, -1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

class pDLoss(nn.Module):
    def __init__(self, n_classes, ignore_index):
        super(pDLoss, self).__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, ignore_mask):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target * ignore_mask)
        y_sum = torch.sum(target * target * ignore_mask)
        z_sum = torch.sum(score * score * ignore_mask)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None):
        ignore_mask = torch.ones_like(target)
        ignore_mask[target == self.ignore_index] = 0
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i], ignore_mask)
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

def find_nonzero_regions(image, m):
    # 将图像转换为布尔类型
    binary_image = image > 0
    # 创建卷积核
    kernel = torch.ones((1, 1, m, m), device=image.device)  # 保持与图像维度一致
    # 对图像进行卷积
    conv_result = F.conv2d(binary_image.float(), kernel, stride=1, padding=0)
    # 筛选出所有值等于 m*m 的坐标
    nonzero_coords = torch.nonzero(conv_result[:] == m * m, as_tuple=False)
    # 提取左上角坐标

    return nonzero_coords

class HCLCHRL_unet(nn.Module):
    def __init__(self, num_channels=4, num_classes=1):
        super(HCLCHRL_unet, self).__init__()
        # self.convdata = nn.Conv2d(num_channels, 3, kernel_size=3, stride=1, padding=1)
        self.inc = inconv(num_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, num_classes)

        self.relu = nn.ReLU()
        self.loss_s = torch.nn.BCELoss()
        self.dice_loss = pDLoss(num_classes, ignore_index=2)
        self.ce_loss = CrossEntropyLoss(ignore_index=2)
        self.projector_1 = projection_head(in_dim=64, hidden_dim=2048, out_dim=128)
        self.projector_3 = projection_head(in_dim=576, hidden_dim=2048, out_dim=128) #64*3*3

    def _generate_samples(self, feature, mask, threshold=0.5, num_samples_per_batch=10, m=1):
        # 创建二进制掩码
        # mask = (mask > threshold).float()
        b, c, h, w = feature.shape

        # 处理没有正样本的批次
        if mask.sum() == 0:
            return torch.tensor([]).to(feature.device)
        # 获取所有有效索引
        # all_indices = mask.nonzero(as_tuple=False)
        all_indices = find_nonzero_regions(mask, m)

        sample_indices = torch.randperm(all_indices.size(0))[:num_samples_per_batch * b]
        selected_indices = all_indices[sample_indices]
        batch_indices, y_indices, x_indices = selected_indices[:, 0], selected_indices[:, 2], selected_indices[:, 3]
        # 使用 unfold 提取所有 [m, m] patch
        # [b, c, h, w] -> [b, c, num_patches_h, num_patches_w, m, m]
        patches_unfolded = feature.unfold(2, m, 1).unfold(3, m, 1)  # 2和3分别是height和width维度
        # 选择对应索引的patch
        patches = patches_unfolded[batch_indices, :, y_indices, x_indices]

        flattened_representation = rearrange(patches, 'n ... -> n (...)')
            # return patches
        if m == 1:
            projector = self.projector_1
        elif m == 3:
            projector = self.projector_3
        projection = projector(flattened_representation)

        return projection

    def _generate_proto(self, feature, mask, threshold=0.5):
        b, c, h, w = feature.shape
        # 创建二进制掩码
        binary_mask = (mask > threshold).float()
        
        # 应用掩码，将特征图与掩码相乘，掩码为0的区域特征将被设置为0
        feat_fore = feature * binary_mask
        
        # 将特征图转换为[N, C, H*W]，其中N是batch size，C是通道数
        feat_fore = feat_fore.view(b, c, -1)
        
        # 计算掩码的非零元素数量，用于后续的归一化
        mask_nonzero_num = binary_mask.view(b, -1).sum(dim=1, keepdim=True)
        
        # 过滤掉掩码为0的区域特征，只保留掩码为1的区域特征
        # 使用掩码的非零元素来索引特征图
        feat_fore = feat_fore * (mask_nonzero_num > 0).float().view(b, 1, -1)
        
        # 聚合特征，这里使用平均池化作为示例
        proto = feat_fore.sum(dim=2, keepdim=True) / mask_nonzero_num
        
        # 归一化原型，使其在通道维度上的L2范数为1
        proto = proto / (proto.norm(dim=1, p=2, keepdim=True) + 1e-5)
        
        return proto

    def _contrastive(self, positive, negative, temperature):
    # 归一化特征向量
        positive = F.normalize(positive, dim=1)
        negative = F.normalize(negative, dim=1)

        samples = torch.cat((positive,negative),dim=0) 
        # 计算所有正样本与所有负样本之间的余弦相似度
        dot_product = torch.matmul(positive, samples.t()) / temperature  # [num_positives, num_negatives]

        # 应用 log-sum-exp 技巧进行数值稳定的 softmax 计算
        logits_max = torch.max(dot_product, dim=1, keepdim=True)[0] #正样本间相似性
        logits = (dot_product - logits_max).detach()  # 从每行减去最大值以增加数值稳定性

        # 生成标签，假设positive样本数量和negative样本数量相同
        mask  = torch.zeros(positive.size(0) + negative.size(0), device=positive.device)
        # 计算对应正样本对和负样本对的掩码
        mask[:positive.size(0)] = 1

        # 将mask的每一行扩展到n*(n+n)
        # pos_mask = mask.repeat_interleave(positive.size(0), dim=1)
        pos_mask = mask.unsqueeze(0).repeat(positive.size(0), 1)
        neg_mask = 1 - pos_mask

        # 创建一个全1的掩码，然后在对角线上置0
        logits_mask = torch.ones_like(pos_mask).scatter_(1, torch.arange(positive.size(0)).view(-1, 1).cuda(), 0)
        # 通过logits_mask调整mask
        mask = pos_mask * logits_mask

        # 计算负logits的指数并应用负样本掩码(负样本对的exp)
        neg_logits = torch.exp(logits) * neg_mask
        # 求负logits的总和(负样本对的exp.sum)
        neg_logits = neg_logits.sum(1, keepdim=True)

        # 计算所有位置上logits的指数(正+负样本对的exp)
        exp_logits = torch.exp(logits)

        # 计算log概率（每一个样本对相似度-正相似*所有负的和）
        log_prob = logits - torch.log(exp_logits + neg_logits)

        # 计算正样本的平均log概率（所有正样本对相似度-正相似*所有负的和）
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # 计算最终的损失
        loss = - mean_log_prob_pos
        # 对损失求平均
        loss = loss.mean()

        # 返回损失
        return loss

    def _correlation(self, features, fore, back):
        feat_fore = features * fore  # 应用掩码，将特征图与掩码相乘
        feat_back= features * back  # 应用掩码，将特征图与掩码相乘
                # 定义一个很小的常数，用于防止除以零
        query_feat = features
        eps = 1e-5

        # query_feat 和 support_feat 是输入的特征图，它们的形状通常是 (batch_size, channels, height, width)
        bsz, ch, hb, wb = feat_fore.size()  # 获取support特征图的尺寸
        feat_fore = feat_fore.view(bsz, ch, -1)  # 将support特征图展平为 (batch_size, channels, height*width)
        feat_back = feat_back.view(bsz, ch, -1)  # 将support特征图展平为 (batch_size, channels, height*width)
        # fore_mask = rearrange(fore, 'n c h w -> n c (h w)')
        # back_mask = rearrange(back, 'n c h w -> n c (h w)')


        # 对support特征图进行归一化，使其在通道维度上的L2范数为1
        feat_fore = feat_fore / (feat_fore.norm(dim=1, p=2, keepdim=True) + eps)
        feat_back = feat_back / (feat_back.norm(dim=1, p=2, keepdim=True) + eps)


        # 使用torch.masked_select选择非零元素
        # selected_features_fore = torch.masked_select(features, fore > 0)
        # 计算非零元素的最大值或平均值
        # feat_fore  = selected_features_fore.max(dim=2)[0]
        # feat_fore  = feat_fore.mean(dim=2)

        # 使用torch.masked_select选择非零元素
        # selected_features_back = torch.masked_select(features, back > 0)
        # 计算非零元素的最大值或平均值
        # feat_back = selected_features_back .max(dim=2)[0]
        # feat_back = feat_back.mean(dim=2)

        bsz, ch, ha, wa = query_feat.size()  # 获取query特征图的尺寸
        query_feat = query_feat.view(bsz, ch, -1)  # 将query特征图展平为 (batch_size, channels, height*width)
        # 对query特征图进行归一化，使其在通道维度上的L2范数为1
        query_feat = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + eps)

        # 将掩码扩展到与特征图相同的通道数
        fore_mask = fore.view(bsz, 1, -1).expand(-1, ch, -1)
        back_mask = back.view(bsz, 1, -1).expand(-1, ch, -1)
        # 计算非零区域的平均特征
        feat_fore = (feat_fore ).sum(dim=2) / (fore_mask.sum(dim=2) + eps)  # 避免除以零
        feat_back = (feat_back ).sum(dim=2) / (back_mask.sum(dim=2) + eps)  # 避免除以零
        feat_fore = feat_fore.unsqueeze(2)
        feat_back = feat_back.unsqueeze(2)

        query_feat = query_feat.float()
        feat_fore = feat_fore.float()
        feat_back = feat_back.float()
        # 使用批量矩阵乘法计算query和support特征图之间的相关性
        corr_fore = torch.bmm(query_feat.transpose(1, 2), feat_fore).view(bsz, ha, wa, 1, 1)
        corr_back = torch.bmm(query_feat.transpose(1, 2), feat_back).view(bsz, ha, wa, 1, 1)

        # 将相关性矩阵中的负值截断为0，因为相关性应该是非负的
        corr_fore = corr_fore.clamp(min=0)
        corr_fore_map = corr_fore.squeeze()
        corr_back = corr_back.clamp(min=0)
        corr_back_map = corr_back.squeeze()

        # self.save_correlation_image(corr_fore_map, 'correlatiom_map.png')
        # 返回堆叠后的相关性矩阵列表
        return corr_fore_map, corr_back_map
    
    # def save_correlation_image(self, correlation_map, filename):
    #     # 将相关性矩阵转换为numpy数组
    #     correlation_map_np = correlation_map.detach().cpu().numpy()
    #     # 取第一个batch的图像
    #     img = correlation_map_np[0]  # 假设我们只取相关性矩阵的第一个通道
    #     # 归一化到[0, 1]区间
    #     img = (img - img.min()) / (img.max() - img.min())
    #     # 保存图像
    #     plt.imsave(filename, img, cmap='hot')  # 使用热图颜色映射

    def forward(self, img, fore, back):
        x1 = self.inc(img)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        d1 = self.up1(x5, x4)
        d2 = self.up2(d1, x3)
        d3 = self.up3(d2, x2)
        out = self.up4(d3, x1)
        output = self.outc(out)
        output = torch.sigmoid(output)

        # loss = F.binary_cross_entropy(output, fore)
        # print('loss_consistency:%f' %(loss))

        # n,c,h,w
        out_x = output.max(dim=2,keepdim=True)[0]
        out_y = output.max(dim=3, keepdim=True)[0]

        print('fore_max:%f' % torch.max(fore))

        # anchor_feats_1= self._generate_samples(out, fore, num_samples_per_batch=1024, m=1)
        pos_feats_1 = self._generate_samples(out, fore, num_samples_per_batch=1024, m=1)
        neg_feats_1 = self._generate_samples(out, back, num_samples_per_batch=2048, m=1)#n*b, c*h*w
        
        # anchor_feats_2= self._generate_samples(out, fore, num_samples_per_batch=256, m=3)
        pos_feats_2= self._generate_samples(out, fore, num_samples_per_batch=256, m=3)
        neg_feats_2= self._generate_samples(out, back, num_samples_per_batch=512, m=3)


        # loss_contrast_1 = self._contrast_nce_fast(anchor_feats_1, pos_feats_1, neg_feats_1, 0.07)
        # loss_contrast_2 = self._contrast_nce_fast(anchor_feats_2, pos_feats_2, neg_feats_2, 0.07)
        loss_contrast_1 = self._contrastive(pos_feats_1, neg_feats_1, 0.07)
        loss_contrast_2 = self._contrastive(pos_feats_2, neg_feats_2, 0.07)

        loss_contrast = 0.5*loss_contrast_1 + 0.5*loss_contrast_2
        print('loss_contrast:%f' %( loss_contrast ))

        corr_fore_map, corr_back_map = self._correlation(out, fore, back)

        return output, out_x, out_y, loss_contrast, corr_fore_map, corr_back_map