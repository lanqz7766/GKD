# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import math
from mmcv.runner import load_checkpoint
import torch.nn as nn
from mmdet.models import build_detector
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
import torch.nn.functional as F
from mmdet.utils import box_ops
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core.bbox import bbox_xyxy_to_cxcywh

from mmdet.utils import get_root_logger, get_device
logger = get_root_logger()

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, freeze=False):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

@DETECTORS.register_module()
class DETR_Distiller(SingleStageDetector):
    r"""Implementation of `Distilling the Knowledge in a Neural Network.
    <https://arxiv.org/abs/1503.02531>`_.
    Args:
        teacher_config (str | dict): Config file path
            or the config object of teacher model.
        teacher_ckpt (str, optional): Checkpoint path of teacher model.
            If left as None, the model will not load any weights.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 teacher_config,
                 teacher_ckpt=None,
                 eval_teacher=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                         pretrained)
        self.eval_teacher = eval_teacher
        # Build teacher model
        if isinstance(teacher_config, str):
            teacher_config = mmcv.Config.fromfile(teacher_config)
        self.teacher_model = build_detector(teacher_config['model'])
        if teacher_ckpt is not None:
            load_checkpoint(
                self.teacher_model, teacher_ckpt, map_location='cpu')
        self.random_refpoints_xy = True
        self.scalar = 3 #5
        self.random_query_number = 300##
        self.embed_dims = 256
        self.random_query_embedding = nn.Embedding(self.random_query_number,
                                                self.embed_dims * 2) ##
        self.mlp = MLP(4, 64, 256, freeze=False)

        self.reference_points = nn.Linear(self.embed_dims, 2)
        if self.random_refpoints_xy:
            # import ipdb; ipdb.set_trace()
            self.reference_points.weight.data[:, :2].uniform_(0,1)
            self.reference_points.weight.data[:, :2] = inverse_sigmoid(self.reference_points.weight.data[:, :2])
            self.reference_points.weight.data[:, :2].requires_grad = False

        self.T = 10
        self.num_classes = 3
        self.with_aux_refpoints = True
        self.with_random_refpoints = True
        self.label_noise_scale = 0 #0.2
        self.box_noise_scale = 0 #0.4
        self.hidden_dim = 256
        self.label_enc = nn.Embedding(self.num_classes + 1, self.hidden_dim - 1)  # # for indicator


    # def prepare_for_dn(dn_args, tgt_weight, embedweight, batch_size, training, num_queries, num_classes, hidden_dim, label_enc):
    #     """
    #     The major difference from DN-DAB-DETR is that the author process pattern embedding pattern embedding in its detector
    #     forward function and use learnable tgt embedding, so we change this function a little bit.
    #     :param dn_args: targets, scalar, label_noise_scale, box_noise_scale, num_patterns
    #     :param tgt_weight: use learnbal tgt in dab deformable detr
    #     :param embedweight: positional anchor queries
    #     :param batch_size: bs
    #     :param training: if it is training or inference
    #     :param num_queries: number of queires
    #     :param num_classes: number of classes
    #     :param hidden_dim: transformer hidden dim
    #     :param label_enc: encode labels in dn
    #     :return:
    #     """

    #     if training:
    #         targets, scalar, label_noise_scale, box_noise_scale, num_patterns = dn_args
    #     else:
    #         num_patterns = dn_args

    #     indicator0 = torch.zeros([num_queries, 1]).cuda()
    #     # sometimes the target is empty, add a zero part of label_enc to avoid unused parameters
    #     tgt = torch.cat([tgt_weight, indicator0], dim=1) + label_enc.weight[0][0]*torch.tensor(0).cuda()
    #     refpoint_emb = embedweight
    #     if training:
    #         known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
    #         know_idx = [torch.nonzero(t) for t in known]
    #         known_num = [sum(k) for k in known]
    #         # you can uncomment this to use fix number of dn queries
    #         # if int(max(known_num))>0:
    #         #     scalar=scalar//int(max(known_num))

    #         # can be modified to selectively denosie some label or boxes; also known label prediction
    #         unmask_bbox = unmask_label = torch.cat(known)
    #         labels = torch.cat([t['labels'] for t in targets])
    #         boxes = torch.cat([t['boxes'] for t in targets])
    #         batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])

    #         known_indice = torch.nonzero(unmask_label + unmask_bbox)
    #         known_indice = known_indice.view(-1)

    #         # add noise
    #         known_indice = known_indice.repeat(scalar, 1).view(-1)
    #         known_labels = labels.repeat(scalar, 1).view(-1)
    #         known_bid = batch_idx.repeat(scalar, 1).view(-1)
    #         known_bboxs = boxes.repeat(scalar, 1)
    #         known_labels_expaned = known_labels.clone()
    #         known_bbox_expand = known_bboxs.clone()

    #         # noise on the label
    #         if label_noise_scale > 0:
    #             p = torch.rand_like(known_labels_expaned.float())
    #             chosen_indice = torch.nonzero(p < (label_noise_scale)).view(-1)  # usually half of bbox noise
    #             new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
    #             known_labels_expaned.scatter_(0, chosen_indice, new_label)
    #         # noise on the box
    #         if box_noise_scale > 0:
    #             diff = torch.zeros_like(known_bbox_expand)
    #             diff[:, :2] = known_bbox_expand[:, 2:] / 2
    #             diff[:, 2:] = known_bbox_expand[:, 2:]
    #             known_bbox_expand += torch.mul((torch.rand_like(known_bbox_expand) * 2 - 1.0),
    #                                            diff).cuda() * box_noise_scale
    #             known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)

    #         m = known_labels_expaned.long().to('cuda')
    #         input_label_embed = label_enc(m)
    #         # add dn part indicator
    #         indicator1 = torch.ones([input_label_embed.shape[0], 1]).cuda()
    #         input_label_embed = torch.cat([input_label_embed, indicator1], dim=1)
    #         input_bbox_embed = inverse_sigmoid(known_bbox_expand)
    #         single_pad = int(max(known_num))
    #         pad_size = int(single_pad * scalar)
    #         padding_label = torch.zeros(pad_size, hidden_dim).cuda()
    #         padding_bbox = torch.zeros(pad_size, 4).cuda()
    #         input_query_label = torch.cat([padding_label, tgt], dim=0).repeat(batch_size, 1, 1)
    #         input_query_bbox = torch.cat([padding_bbox, refpoint_emb], dim=0).repeat(batch_size, 1, 1)
    #         # map in order
    #         map_known_indice = torch.tensor([]).to('cuda')
    #         if len(known_num):
    #             map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
    #             map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()
    #         if len(known_bid):
    #             input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
    #             input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

    #         tgt_size = pad_size + num_queries * num_patterns

    #         attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
    #         # match query cannot see the reconstruct
    #         attn_mask[pad_size:, :pad_size] = True ##
    #         # reconstruct cannot see each other
    #         for i in range(scalar):
    #             if i == 0:
    #                 attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
    #             if i == scalar - 1:
    #                 attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
    #             else:
    #                 attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
    #                 attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
    #         mask_dict = {
    #             'known_indice': torch.as_tensor(known_indice).long(),
    #             'batch_idx': torch.as_tensor(batch_idx).long(),
    #             'map_known_indice': torch.as_tensor(map_known_indice).long(),
    #             'known_lbs_bboxes': (known_labels, known_bboxs),
    #             'know_idx': know_idx,
    #             'pad_size': pad_size
    #         }
    #     else:  # no dn for inference
    #         input_query_label = tgt.repeat(batch_size, 1, 1)
    #         input_query_bbox = refpoint_emb.repeat(batch_size, 1, 1)
    #         attn_mask = None
    #         mask_dict = None

    #     # input_query_label = input_query_label.transpose(0, 1)
    #     # input_query_bbox = input_query_bbox.transpose(0, 1)

    #     return input_query_label, input_query_bbox, attn_mask, mask_dict

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

        bs, C, H, W= img.size()
        random_query_embeds = self.random_query_embedding.weight.detach() ##
        random_query_pos, random_query = torch.split(random_query_embeds, self.embed_dims, dim=1) # 2*(num_query, embed)
        random_query_pos = random_query_pos.unsqueeze(0).expand(bs, -1, -1).detach() # (bs, num_query, embed)
        # random_query = random_query.unsqueeze(0).expand(bs, -1, -1).detach() # (bs, num_query, embed)
        # random_reference_points = self.reference_points(random_query_pos).sigmoid().detach() # (bs, num_query, 2)

        # '''
        # random_reference_points = self.reference_points(random_query_pos)
        # random_query_pos = random_query_pos.unsqueeze(0).expand(bs, -1, -1).detach() # (bs, num_query, embed)
        gt_bboxes_list = []
        for i in range(len(gt_bboxes)):
            img_h, img_w, _ = img_metas[i]['img_shape']
            bboxes = gt_bboxes[i]
            factor = bboxes.new_tensor([img_w, img_h, img_w,
                                        img_h]).unsqueeze(0)
            bboxes_normalized = bboxes / factor
            gt_bboxes_list.append(bboxes_normalized)
        boxes_xyxy = torch.cat(gt_bboxes_list)
        boxes = bbox_xyxy_to_cxcywh(boxes_xyxy)

        # indicator0 = torch.zeros([self.random_query_number, 1]).cuda()
        # sometimes the target is empty, add a zero part of label_enc to avoid unused parameters
        # tgt = torch.cat([tgt_weight, indicator0], dim=1) + label_enc.weight[0][0]*torch.tensor(0).cuda()
        # refpoint_emb = random_query_embeds
        known = [torch.ones_like(t).cuda() for t in gt_labels]
        know_idx = [torch.nonzero(t) for t in known]
        known_num = [sum(k) for k in known]
        # you can uncomment this to use fix number of dn queries
        # if int(max(known_num))>0:
        #     self.scalar=self.scalar//int(max(known_num))
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t for t in gt_labels])
        batch_idx = torch.cat([torch.full_like(t.long(), i) for i, t in enumerate(gt_labels)])

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        # add noise
        known_indice = known_indice.repeat(self.scalar, 1).view(-1)
        known_labels = labels.repeat(self.scalar, 1).view(-1)
        known_bid = batch_idx.repeat(self.scalar, 1).view(-1)
        known_bboxs = boxes.repeat(self.scalar, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        # noise on the label
        if self.label_noise_scale > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (self.label_noise_scale)).view(-1)  # usually half of bbox noise
            new_label = torch.randint_like(chosen_indice, 0, 3)  # randomly put a new one here #num_classes here is 3
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        # noise on the box
        if self.box_noise_scale > 0:
            diff = torch.zeros_like(known_bbox_expand)
            diff[:, :2] = known_bbox_expand[:, 2:] / 2
            diff[:, 2:] = known_bbox_expand[:, 2:]
            known_bbox_expand += torch.mul((torch.rand_like(known_bbox_expand) * 2 - 1.0),
                                           diff).cuda() * self.box_noise_scale
            known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)

        m = known_labels_expaned.long().to('cuda')
        input_label_embed = self.label_enc(m)
        # add dn part indicator
        indicator1 = torch.ones([input_label_embed.shape[0], 1]).cuda()
        input_label_embed = torch.cat([input_label_embed, indicator1], dim=1)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)
        # input_bbox_embed = inverse_sigmoid(known_bbox_expand[:,:2]) ###
        single_pad = int(max(known_num))
        pad_size = int(single_pad * self.scalar)
        padding_label = torch.zeros(pad_size, self.hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 4).cuda()  
        # padding_bbox = torch.zeros(pad_size, 2).cuda() ###
        input_query_label = torch.cat([padding_label, random_query], dim=0).repeat(bs, 1, 1)
        input_query_bbox = torch.cat([padding_bbox], dim=0).repeat(bs, 1, 1)

        # map in order
        map_known_indice = torch.tensor([]).to('cuda')
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(self.scalar)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        # tgt_size = pad_size + num_queries * num_patterns ##
        tgt_size = pad_size
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct
        # attn_mask[pad_size:, :pad_size] = True ##
        # reconstruct cannot see each other
        for i in range(self.scalar):
            if i == 0:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            if i == self.scalar - 1:
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            else:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        mask_dict = {
            'known_indice': torch.as_tensor(known_indice).long(),
            'batch_idx': torch.as_tensor(batch_idx).long(),
            'map_known_indice': torch.as_tensor(map_known_indice).long(),
            'known_lbs_bboxes': (known_labels, known_bboxs),
            'know_idx': know_idx,
            'pad_size': pad_size
        }

        #return input_query_label, input_query_bbox, attn_mask, mask_dict
        dn_query = self.mlp(input_query_bbox)
        dn_random_query_pos = torch.cat([dn_query, random_query_pos], dim=1).clone().detach()

        dn_random_query = input_query_label.clone().detach()

        dn_random_reference_points = self.reference_points(dn_random_query_pos).sigmoid().detach()

        
        # logger.info(f'dn_random_query : {dn_random_query.shape}')
        # logger.info(f'dn_random_query_pos : {dn_random_query_pos.shape}')
        # logger.info(f'dn_random_reference_points : {dn_random_reference_points.shape}')
        # logger.info(f' mask_dict : { mask_dict}')
        # '''
        x = self.extract_feat(img)
        with torch.no_grad():
            self.teacher_model.eval()
            teacher_x = self.teacher_model.extract_feat(img)
            T_teacher_outs, T_random_outs, teacher_query, teacher_query_pos, teacher_reference_points = self.teacher_model.bbox_head.simple_test_teacher_distill(teacher_x, img_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels,\
                         random_query = dn_random_query, random_query_pos = dn_random_query_pos, random_reference_points = dn_random_reference_points)
 

        teacher_query = teacher_query.detach()
        chercher_query_pos = teacher_query_pos.detach()
        teacher_reference_points = teacher_reference_points.detach()
        
        student_loss, S_teacher_outs, S_random_outs = self.bbox_head.forward_train_distill_aux(x, img_metas, gt_bboxes=gt_bboxes, gt_labels=gt_labels, random_query = dn_random_query, random_query_pos = dn_random_query_pos, random_reference_points = dn_random_reference_points, \
            teacher_query=teacher_query, teacher_query_pos = teacher_query_pos, teacher_reference_points = teacher_reference_points)


        T_teacher_outputs_classes, T_teacher_outputs_coords, _, _ = T_teacher_outs #(6, 2, 300, 3), (6, 2, 300, 4)
        T_random_outputs_classes, T_random_outputs_coords, _, _ = T_random_outs 


        S_teacher_outputs_classes, S_teacher_outputs_coords, _, _ = S_teacher_outs #(6, 2, 300, 3), 6, 2, 300, 4)
        S_random_outputs_classes, S_random_outputs_coords, _, _= S_random_outs 
        
        stages = len(S_random_outputs_classes)
        if self.with_aux_refpoints:
            for stage in range(stages):
                aux_weight = T_teacher_outputs_classes[stage].flatten(0, 1)
                aux_weight = aux_weight.sigmoid().max(1)[0].detach()
                student_loss[f'stage{stage}_aux_kd_cls_loss']= self.loss_kl_div(T_teacher_outputs_classes[stage], S_teacher_outputs_classes[stage], aux_weight)
                loss_tmp = self.loss_boxes(T_teacher_outputs_coords[stage], S_teacher_outputs_coords[stage], aux_weight)
                student_loss[f'stage{stage}_kd_auxrf_bbox_loss'] = loss_tmp['loss_bbox']
                student_loss[f'stage{stage}_kd_auxrf_giou_loss'] = loss_tmp['loss_giou']



        if self.with_random_refpoints:
            for stage in range(stages):
                random_weight = T_random_outputs_classes[stage].flatten(0, 1)
                random_weight = random_weight.sigmoid().max(1)[0].detach()
                student_loss[f'stage{stage}_random_kd_cls_loss'] = self.loss_kl_div(T_random_outputs_classes[stage], S_random_outputs_classes[stage], random_weight)
                loss_tmp = self.loss_boxes(T_random_outputs_coords[stage], S_random_outputs_coords[stage], random_weight)
                student_loss[f'stage{stage}_kd_random_bbox_loss'] = loss_tmp['loss_bbox']
                student_loss[f'stage{stage}_kd_random_giou_loss'] = loss_tmp['loss_giou']


        return student_loss

    def loss_kl_div(self, pred, soft_label, weight=None):
        assert pred.size() == soft_label.size()
       
        if len(pred.shape) > 2:
            pred = pred.flatten(start_dim=0, end_dim=-2)
            soft_label = soft_label.flatten(start_dim=0, end_dim=-2)
        
        target = F.softmax(soft_label / self.T, dim=1)
        target = target.detach()
        if weight is not None:
            kd_loss = F.kl_div(F.log_softmax(pred / self.T, dim=1), target, reduction='none').mean(1) * (self.T * self.T) 
            kd_loss = sum(kd_loss * weight)
        else:
            kd_loss = F.kl_div(F.log_softmax(pred / self.T, dim=1), target, reduction='none').mean(1) * (self.T * self.T)
            kd_loss = kd_loss.mean()

        return kd_loss

    def loss_boxes(self, pred, soft_label, weight=None):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """

        losses = {}
        if len(pred.shape) == 3:
            pred = pred.flatten(0, 1)
            soft_label = soft_label.flatten(0, 1)
        loss_bbox = F.l1_loss(pred, soft_label, reduction='none').mean(1)
        losses['loss_bbox'] = 5*sum(loss_bbox * weight) / weight.sum()

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(pred),
            box_ops.box_cxcywh_to_xyxy(soft_label)))
        losses['loss_giou'] = 2*sum(loss_giou * weight) / weight.sum()

        return losses


    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        warnings.warn('Warning! MultiheadAttention in DETR does not '
                      'support flops computation! Do not use the '
                      'results in your papers!')

        batch_size, _, height, width = img.shape
        dummy_img_metas = [
            dict(
                batch_input_shape=(height, width),
                img_shape=(height, width, 3)) for _ in range(batch_size)
        ]
        x = self.extract_feat(img)
        outs = self.bbox_head(x, dummy_img_metas)
        return outs

    def onnx_export(self, img, img_metas):
        """Test function for exporting to ONNX, without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        # forward of this head requires img_metas
        outs = self.bbox_head.forward_onnx(x, img_metas)
        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape

        det_bboxes, det_labels = self.bbox_head.onnx_export(*outs, img_metas)

        return det_bboxes, det_labels

    def cuda(self, device=None):
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to cuda when calling cuda function."""
        self.teacher_model.cuda(device=device)
        return super().cuda(device=device)

    def train(self, mode=True):
        """Set the same train mode for teacher and student model."""
        if self.eval_teacher:
            self.teacher_model.train(False)
        else:
            self.teacher_model.train(mode)
        super().train(mode)

    def __setattr__(self, name, value):
        """Set attribute, i.e. self.name = value
        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'teacher_model':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)
