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

from mmdet.utils import collect_env, get_root_logger


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
        super(KDAttention).__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                         pretrained, init_cfg)
        self.eval_teacher = eval_teacher
        # Build teacher model
        if isinstance(teacher_config, str):
            teacher_config = mmcv.Config.fromfile(teacher_config)
        self.teacher_model = build_detector(teacher_config['model'])
        if teacher_ckpt is not None:
            load_checkpoint(
                self.teacher_model, teacher_ckpt, map_location='cpu')

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
        x = self.extract_feat(img)
        with torch.no_grad():
            teacher_x = self.teacher_model.extract_feat(img)
            out_teacher = self.teacher_model.bbox_head(teacher_x, img_metas)
#        losses = self.bbox_head.forward_train(x, out_teacher, img_metas,
#                                              gt_bboxes, gt_labels,
#                                              gt_bboxes_ignore)

        losses = self.bbox_head.forward_train(x, img_metas,
                                              gt_bboxes, gt_labels,
                                              gt_bboxes_ignore)

        outs = self.bbox_head(x, img_metas)

        T_teacher_outputs_classes, T_teacher_outputs_coords, _, _ = T_teacher_outs #(6, 2, 300, 3), (6, 2, 300, 4)

        S_teacher_outputs_classes, S_teacher_outputs_coords, _, _ = S_teacher_outs #(6, 2, 300, 3), 6, 2, 300, 4)

       logger.info(f"T_teacher_outputs_classes and T_teacher_outputs_coords shape is: {T_teacher_outputs_classes.size(), T_teacher_outputs_coords.size()}")
       logger.info(f"S_teacher_outputs_classes and S_teacher_outputs_coords shape is: {T_teacher_outputs_classes.size(), T_teacher_outputs_coords.size()}")
#        weight = torch.exp(-losses['loss_cls'][0] - losses ['loss_bbox'][0])
#        logger.info(f"weight values are: {weight}")
#        with torch.no_grad():
#            teacher_x = self.teacher_model.extract_feat(img)
#            out_teacher = self.teacher_model.bbox_head(teacher_x)

#        cls_quality_score, bbox_pred = self.teacher_model.bbox_head.forward(teacher_x)
#        loss_test = self.teacher_model.bbox_head.loss(cls_quality_score,
#                                                      bbox_pred,
#                                                      gt_bboxes, gt_labels,
#                                                      img_metas, gt_bboxes_ignore)


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
