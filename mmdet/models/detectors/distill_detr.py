# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .detr import DETR


@DETECTORS.register_module()
class Distill_DETR(DETR):

    def __init__(self, *args, **kwargs):
        super(DETR, self).__init__(*args, **kwargs)

    def forward_train_aux(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      random_query = None,
                      random_query_pos = None,
                      random_reference_points = None,
                      teacher_query=None,
                      teacher_query_pos=None,
                      teacher_reference_points=None):
    
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
    
        x = self.extract_feat(img)
        losses, teacher_outs, random_outs = self.bbox_head.forward_train_distill_aux(x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore, 
                                              random_query, random_query_pos, random_reference_points ,teacher_query, teacher_query_pos, teacher_reference_points)
        return losses, teacher_outs, random_outs, x