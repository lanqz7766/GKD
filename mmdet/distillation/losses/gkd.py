import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from mmcv.cnn import constant_init, kaiming_init
from ..builder import DISTILL_LOSSES
import cv2
from collections import OrderedDict

# from mmdet.utils import get_root_logger
# logger = get_root_logger()

@DISTILL_LOSSES.register_module()
class GKD_FeatureLoss(nn.Module):

    """PyTorch version of `Focal and Global Knowledge Distillation for Detectors`
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        temp (float, optional): Temperature coefficient. Defaults to 0.5.
        name (str): the loss name of the layer
        alpha_fgd (float, optional): Weight of fg_loss. Defaults to 0.001
        beta_fgd (float, optional): Weight of bg_loss. Defaults to 0.0005
        gamma_fgd (float, optional): Weight of mask_loss. Defaults to 0.001
        lambda_fgd (float, optional): Weight of relation_loss. Defaults to 0.000005
    """
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 name,
                 temp=0.5,
                 alpha_fgd=0.001,
                 beta_fgd=0.05,
                 gamma_fgd=0.001,
                 lambda_fgd=0.000005,
                 ):
        super(GKD_FeatureLoss, self).__init__()
        self.temp = temp
        self.alpha_fgd = alpha_fgd
        self.beta_fgd = beta_fgd
        self.gamma_fgd = gamma_fgd
        self.lambda_fgd = lambda_fgd

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None
        
        self.conv_mask_s = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.conv_mask_t = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.channel_add_conv_s = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels//2, kernel_size=1),
            nn.LayerNorm([teacher_channels//2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels//2, teacher_channels, kernel_size=1))
        self.channel_add_conv_t = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels//2, kernel_size=1),
            nn.LayerNorm([teacher_channels//2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels//2, teacher_channels, kernel_size=1))

        self.reset_parameters()


    def forward(self,
                preds_S,
                preds_T,
                cam_masks_T,
                cam_masks_S,
                gt_bboxes,
                img_metas):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
            gt_bboxes(tuple): Bs*[nt*4], pixel decimal: (tl_x, tl_y, br_x, br_y)
            img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:],'the output dim of teacher and student differ'

        if self.align is not None:
            preds_S = self.align(preds_S)
        
        N,C,H,W = preds_S.shape

        S_attention_t, C_attention_t = self.get_attention(preds_T, self.temp)
        # S_attention_s, C_attention_s = self.get_attention(preds_S, self.temp)
        Mask_fg = torch.zeros_like(S_attention_t)
        # Mask_bg = torch.ones_like(S_attention_t)
        wmin,wmax,hmin,hmax = [],[],[],[]
        big_map = torch.zeros(N, H*2, W*2).cuda()
        for i in range(N):
            new_boxxes = torch.ones_like(gt_bboxes[i])
            new_boxxes[:, 0] = gt_bboxes[i][:, 0]/img_metas[i]['img_shape'][1]*W
            new_boxxes[:, 2] = gt_bboxes[i][:, 2]/img_metas[i]['img_shape'][1]*W
            new_boxxes[:, 1] = gt_bboxes[i][:, 1]/img_metas[i]['img_shape'][0]*H
            new_boxxes[:, 3] = gt_bboxes[i][:, 3]/img_metas[i]['img_shape'][0]*H

            wmin.append(torch.floor(new_boxxes[:, 0]).int())
            wmax.append(torch.ceil(new_boxxes[:, 2]).int())
            hmin.append(torch.floor(new_boxxes[:, 1]).int())
            hmax.append(torch.ceil(new_boxxes[:, 3]).int())
            for j in range(len(gt_bboxes[i])):
                w = (wmax[i][j]-wmin[i][j]).int()
                h = (hmax[i][j]-hmin[i][j]).int()
                big_map[i][2*hmin[i][j]:2*hmax[i][j], 2*wmin[i][j]:2*wmax[i][j]] = \
                        torch.maximum(big_map[i][2*hmin[i][j]:2*hmax[i][j], 2*wmin[i][j]:2*wmax[i][j]], self.calculate_gaussian(w, h))
            gap_W = np.floor((2*W-W)/2).astype(int)
            gap_H = np.floor((2*H-H)/2).astype(int)
            Mask_fg[i] = big_map[i][gap_H:gap_H+H, gap_W:gap_W+W]

            # for j in range(len(gt_bboxes[i])):
            #     Mask_fg[i][hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1] = \
            #             torch.maximum(Mask_fg[i][hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1], area[0][j])

            # Mask_bg[i] = torch.where(Mask_fg[i]>0, 0, 1)
            # if torch.sum(Mask_bg[i]):
                # Mask_bg[i] /= torch.sum(Mask_bg[i])
        Mask_cam_T = torch.squeeze(cam_masks_T).cuda()
        Mask_cam_S = torch.squeeze(cam_masks_S).cuda()
        # fg_loss, bg_loss = self.get_fea_loss(preds_S, preds_T, Mask_fg, Mask_bg, Mask_cam_T, Mask_cam_S,
                           # C_attention_s, C_attention_t, S_attention_s, S_attention_t)
        # mask_loss = self.get_mask_loss(C_attention_s, C_attention_t, S_attention_s, S_attention_t)
        # rela_loss = self.get_rela_loss(preds_S, preds_T)
        cam_mask_loss = self.get_cam_mask_loss(Mask_cam_T, Mask_cam_S, preds_S, self.temp)


        # loss = self.alpha_fgd * fg_loss + self.beta_fgd * cam_mask_loss \
        #        + self.gamma_fgd * mask_loss + self.lambda_fgd * rela_loss
        loss = self.beta_fgd * cam_mask_loss
            
        return loss


    def calculate_gaussian(self, old_w, old_h, expand_factor = 2):
        w = expand_factor*old_w
        h = expand_factor*old_h
        cx = w / 2
        cy = h / 2
        if cx == 0:
            cx += 1
        if cy == 0:
            cy += 1
        x0 = cx.repeat(1, w)
        y0 = cy.repeat(h, 1)
        x = torch.arange(w).cuda()
        y = torch.unsqueeze(torch.arange(h), dim=1).cuda()
        gaussian_mask = torch.exp(-0.5*((x-x0)/cx)**2) * torch.exp(-0.5*((y-y0)/cy)**2)
        bbox = torch.ones((old_h,old_w)).cuda()
        gap_w = torch.floor((w-old_w)/2).int()
        gap_h = torch.floor((h-old_h)/2).int()
        gaussian_mask[gap_h:gap_h+old_h, gap_w:gap_w+old_w] = bbox
        gaussian_mask = 1/w/h*gaussian_mask.double().cuda()
        return gaussian_mask


    def get_attention(self, preds, temp):
        """ preds: Bs*C*W*H """
        N, C, H, W= preds.shape

        value = torch.abs(preds)
        # Bs*W*H
        fea_map = value.mean(axis=1, keepdim=True)
        S_attention = (H * W * F.softmax((fea_map/temp).view(N,-1), dim=1)).view(N, H, W)

        # Bs*C
        channel_map = value.mean(axis=2,keepdim=False).mean(axis=2,keepdim=False)
        C_attention = C * F.softmax(channel_map/temp, dim=1)

        return S_attention, C_attention


    # def get_fea_loss(self, preds_S, preds_T, Mask_fg, Mask_bg, Mask_cam_T, Mask_cam_S, C_s, C_t, S_s, S_t):
    #     loss_mse = nn.MSELoss(reduction='sum')
        
    #     Mask_fg = Mask_fg.unsqueeze(dim=1)
    #     Mask_bg = Mask_bg.unsqueeze(dim=1)
    #     Mask_cam_T = Mask_cam_T.unsqueeze(dim=1)
    #     Mask_cam_S = Mask_cam_S.unsqueeze(dim=1)

    #     C_t = C_t.unsqueeze(dim=-1)
    #     C_t = C_t.unsqueeze(dim=-1)

    #     S_t = S_t.unsqueeze(dim=1)

    #     fea_t= torch.mul(preds_T, torch.sqrt(S_t))
    #     fea_t = torch.mul(fea_t, torch.sqrt(C_t))
    #     fg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_fg))
    #     fg_fea_t = torch.mul(fg_fea_t, torch.sqrt(Mask_cam_T))
    #     bg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_bg))

    #     fea_s = torch.mul(preds_S, torch.sqrt(S_t))
    #     fea_s = torch.mul(fea_s, torch.sqrt(C_t))
    #     fg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_fg))
    #     fg_fea_s = torch.mul(fg_fea_s, torch.sqrt(Mask_cam_S))
    #     bg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_bg))

    #     fg_loss = loss_mse(fg_fea_s, fg_fea_t)/len(Mask_fg)
    #     bg_loss = loss_mse(bg_fea_s, bg_fea_t)/len(Mask_bg)

    #     return fg_loss, bg_loss


    # def get_mask_loss(self, C_s, C_t, S_s, S_t):

    #     mask_loss = torch.sum(torch.abs((C_s-C_t)))/len(C_s) + torch.sum(torch.abs((S_s-S_t)))/len(S_s)

    #     return mask_loss

    def get_cam_mask_loss(self, Mask_cam_T, Mask_cam_S, preds_S, temp):
        # loss_mse = nn.MSELoss(reduction='sum')
        # Mask_fg = Mask_fg.unsqueeze(dim=1)
        N,C,H,W = preds_S.shape
        Mask_cam_T = Mask_cam_T.unsqueeze(dim=1)
        Mask_cam_S = Mask_cam_S.unsqueeze(dim=1)

        # Mask_cam_T = (H * W * F.softmax((Mask_cam_T/temp).view(N,-1), dim=1)).view(N, 1, H, W)
        # Mask_cam_S = (H * W * F.softmax((Mask_cam_S/temp).view(N,-1), dim=1)).view(N, 1, H, W)

        cam_mask_loss = torch.sum(torch.abs(Mask_cam_T - Mask_cam_S))/(len(Mask_cam_S)*H*W)
        # cam_mask_loss = loss_mse(Mask_cam_T, Mask_cam_S)/(len(Mask_cam_S)*H*W)

        return cam_mask_loss
        
    # def get_cam_mask_loss(self, Mask_cam_T, Mask_cam_S, preds_S):

    #     Mask_fg = Mask_fg.unsqueeze(dim=1)
    #     Mask_cam_T = Mask_cam_T.unsqueeze(dim=1)
    #     Mask_cam_S = Mask_cam_S.unsqueeze(dim=1)

    #     weighted_mask_cam_T = torch.mul(Mask_cam_T, torch.sqrt(Mask_fg))
    #     weighted_mask_cam_S = torch.mul(Mask_cam_S, torch.sqrt(Mask_fg))
    #     cam_mask_loss = torch.sum(torch.abs(weighted_mask_cam_T - weighted_mask_cam_S))/len(weighted_mask_cam_S)

    #     return cam_mask_loss
     
    
    def spatial_pool(self, x, in_type):
        batch, channel, width, height = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        if in_type == 0:
            context_mask = self.conv_mask_s(x)
        else:
            context_mask = self.conv_mask_t(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = F.softmax(context_mask, dim=2)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context


    def get_rela_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')

        context_s = self.spatial_pool(preds_S, 0)
        context_t = self.spatial_pool(preds_T, 1)

        out_s = preds_S
        out_t = preds_T

        channel_add_s = self.channel_add_conv_s(context_s)
        out_s = out_s + channel_add_s

        channel_add_t = self.channel_add_conv_t(context_t)
        out_t = out_t + channel_add_t

        rela_loss = loss_mse(out_s, out_t)/len(out_s)
        
        return rela_loss


    def last_zero_init(self, m):
        if isinstance(m, nn.Sequential):
            constant_init(m[-1], val=0)
        else:
            constant_init(m, val=0)

    
    def reset_parameters(self):
        kaiming_init(self.conv_mask_s, mode='fan_in')
        kaiming_init(self.conv_mask_t, mode='fan_in')
        self.conv_mask_s.inited = True
        self.conv_mask_t.inited = True

        self.last_zero_init(self.channel_add_conv_s)
        self.last_zero_init(self.channel_add_conv_t)