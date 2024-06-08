import torch.nn as nn
import torch.nn.functional as F
import torch
import mmcv
import os.path
from functools import partial
from mmdet.models.detectors.base import BaseDetector
from mmdet.models import build_detector
from mmcv.runner import  load_checkpoint, _load_checkpoint, load_state_dict
from mmcv import Config, DictAction
from ..builder import DISTILLER, build_distill_loss
from collections import OrderedDict
import numpy as np
import cv2
from typing import Callable, List, Tuple
from mmdet.utils import get_root_logger, get_device
from mmdet.utils.det_cam_visualizer import (DetAblationLayer, Student_DetCAMModel,
                                            DetBoxScoreTarget, DetCAMModel,
                                            DetCAMVisualizer, EigenCAM, BaseCAM,
                                            FeatmapAM, reshape_transform, GradCAM)

try:
    from pytorch_grad_cam import (AblationCAM, EigenGradCAM, GradCAMPlusPlus, LayerCAM, XGradCAM)
except ImportError:
    raise ImportError('Please run `pip install "grad-cam"` to install '
                      '3rd party package pytorch_grad_cam.')

logger = get_root_logger()

GRAD_FREE_METHOD_MAP = {
    'ablationcam': AblationCAM,
    'eigencam': EigenCAM,
    # 'scorecam': ScoreCAM, # consumes too much memory
    'featmapam': FeatmapAM
}

GRAD_BASE_METHOD_MAP = {
    'gradcam': GradCAM,
    'gradcam++': GradCAMPlusPlus,
    'xgradcam': XGradCAM,
    'eigengradcam': EigenGradCAM,
    'layercam': LayerCAM,
    'basecam': BaseCAM
}

ALL_METHODS = list(GRAD_FREE_METHOD_MAP.keys() | GRAD_BASE_METHOD_MAP.keys())

def init_model_cam(cfg,
                   detector,
                   checkpoint,
                   target_layers_, 
                   max_shape = -1, 
                   method = 'gradcam', 
                   score_thr = 0.3, 
                   device = 'cuda:0',
                   student = False):
    if student == False:
        model = DetCAMModel(
            cfg, checkpoint, score_thr, device=device)
    else:
        model = Student_DetCAMModel(
            cfg, checkpoint, detector, score_thr, device=device)

    # if args.preview_model:
    #     print(model.detector)
    #     print('\n Please remove `--preview-model` to get the CAM.')
    #     return

    target_layers = []
    for target_layer in target_layers_:
        try:
            target_layers.append(eval(f'model.detector.{target_layer}'))
        except Exception as e:
            print(model.detector)
            raise RuntimeError('layer does not exist', e)

    extra_params = {
        'batch_size': 1,
        'ablation_layer': DetAblationLayer(),
        'ratio_channels_to_ablate': 0.5
    }

    if method in GRAD_BASE_METHOD_MAP:
        method_class = GRAD_BASE_METHOD_MAP[method]
        is_need_grad = True
        no_norm_in_bbox = False
        assert no_norm_in_bbox is False, 'If not norm in bbox, the ' \
                                              'visualization result ' \
                                              'may not be reasonable.'
    else:
        method_class = GRAD_FREE_METHOD_MAP[method]
        is_need_grad = False

    max_shape = max_shape
    if not isinstance(max_shape, list):
        max_shape = [max_shape]
    assert len(max_shape) == 1 or len(max_shape) == 2

    det_cam_visualizer = DetCAMVisualizer(
        method_class,
        model,
        target_layers,
        reshape_transform=partial(
            reshape_transform, max_shape=max_shape, is_need_grad=is_need_grad),
        is_need_grad=is_need_grad,
        extra_params=extra_params)
    return model, det_cam_visualizer


@DISTILLER.register_module()
class Two_Stage_GKD_DetectionDistiller(BaseDetector):
    """Base distiller for detectors.

    It typically consists of teacher_model and student_model.
    """
    def __init__(self,
                 teacher_cfg,
                 student_cfg,
                 distill_cfg=None,
                 teacher_pretrained=None,
                 init_student=False):

        super(Two_Stage_GKD_DetectionDistiller, self).__init__()
        
        self.teacher = build_detector(teacher_cfg.model,
                                        train_cfg=teacher_cfg.get('train_cfg'),
                                        test_cfg=teacher_cfg.get('test_cfg'))
        self.init_weights_teacher(teacher_pretrained)
        self.teacher.eval()

        self.student= build_detector(student_cfg.model,
                                        train_cfg=student_cfg.get('train_cfg'),
                                        test_cfg=student_cfg.get('test_cfg'))
        self.student.init_weights() ##added
        if init_student:
            t_checkpoint = _load_checkpoint(teacher_pretrained, map_location= 'cpu')
            all_name = []
            for name, v in t_checkpoint["state_dict"].items():
                if name.startswith("backbone."):
                    continue
                else:
                    all_name.append((name, v))

            state_dict = OrderedDict(all_name)
            load_state_dict(self.student, state_dict)

        self.distill_losses = nn.ModuleDict()
        self.distill_cfg = distill_cfg

        student_modules = dict(self.student.named_modules())
        teacher_modules = dict(self.teacher.named_modules())

        self.teacher_cam_model, self.teacher_det_cam_visualizer = init_model_cam(cfg = teacher_cfg,
                                                         detector = None,
                                                         checkpoint = teacher_pretrained,
                                                         target_layers_ = [
                                                         # 'neck.fpn_convs[4].conv', 
                                                         'neck.fpn_convs[3].conv',
                                                         'neck.fpn_convs[2].conv',
                                                         'neck.fpn_convs[1].conv',
                                                         'neck.fpn_convs[0].conv'
                                                         ], 
                                                         max_shape = -1, 
                                                         method = 'gradcam', 
                                                         score_thr = 0.3,
                                                         device = 'cpu',
                                                         student = False)
        self.student_cam_model, self.student_det_cam_visualizer = init_model_cam(cfg = teacher_cfg,
                                                         detector = self.student,
                                                         checkpoint = teacher_pretrained,
                                                         target_layers_ = [
                                                         # 'neck.fpn_convs[4].conv', 
                                                         'neck.fpn_convs[3].conv',
                                                         'neck.fpn_convs[2].conv',
                                                         'neck.fpn_convs[1].conv',
                                                         'neck.fpn_convs[0].conv'
                                                         ], 
                                                         max_shape = -1, 
                                                         method = 'gradcam', 
                                                         score_thr = 0.3,
                                                         device = 'cpu',
                                                         student = True)

        def regitster_hooks(student_module,teacher_module):
            def hook_teacher_forward(module, input, output):

                    self.register_buffer(teacher_module,output)
                
            def hook_student_forward(module, input, output):

                    self.register_buffer(student_module,output)
            return hook_teacher_forward,hook_student_forward
        
        for item_loc in distill_cfg:
            
            student_module = 'student_' + item_loc.student_module.replace('.','_')
            teacher_module = 'teacher_' + item_loc.teacher_module.replace('.','_')

            self.register_buffer(student_module,None)
            self.register_buffer(teacher_module,None)

            hook_teacher_forward,hook_student_forward = regitster_hooks(student_module ,teacher_module )
            teacher_modules[item_loc.teacher_module].register_forward_hook(hook_teacher_forward)
            student_modules[item_loc.student_module].register_forward_hook(hook_student_forward)

            for item_loss in item_loc.methods:
                loss_name = item_loss.name
                self.distill_losses[loss_name] = build_distill_loss(item_loss)
    def base_parameters(self):
        return nn.ModuleList([self.student,self.distill_losses])


    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self.student, 'neck') and self.student.neck is not None

    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self.student, 'roi_head') and self.student.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_bbox)
                or (hasattr(self.student, 'bbox_head') and self.student.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_mask)
                or (hasattr(self.student, 'mask_head') and self.student.mask_head is not None))

    def init_weights_teacher(self, path=None):
        """Load the pretrained model in teacher detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        checkpoint = load_checkpoint(self.teacher, path, map_location='cpu')



    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, **kwargs):

        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components(student's losses and distiller's losses).
        """

        device_ = f'cuda:{img.get_device()}'
        self.teacher_cam_model.device = device_
        self.teacher_det_cam_visualizer.cam.model.device  = device_
        self.student_cam_model.device = device_
        self.student_det_cam_visualizer.cam.model.device = device_
        current_weight = self.student.state_dict()
        self.student_cam_model.update_model_weights(current_weight)

        with torch.no_grad():
            self.teacher.eval()
            feat = self.teacher.extract_feat(img)

        N, C, H, W = img.shape
        images = []
        teacher_cam_masks = []
        student_cam_masks = []
        topk=10
        method = 'gradcam'
        # logger.info(f"target_layer is: {self.det_cam_visualizer.target_layers}")
        for i in range(N):
            images.append(img_metas[i].get('filename'))
        for image_path in images:
            image = cv2.imread(image_path)
            self.teacher_cam_model.set_input_data(image)
            self.student_cam_model.set_input_data(image)
            teacher_result = self.teacher_cam_model()[0]
            student_result = self.student_cam_model()[0]

            teacher_bboxes = teacher_result['bboxes'][..., :4]
            teacher_scores = teacher_result['bboxes'][..., 4]
            teacher_labels = teacher_result['labels']
            teacher_segms = teacher_result['segms']

            student_bboxes = student_result['bboxes'][..., :4]
            student_scores = student_result['bboxes'][..., 4]
            student_labels = student_result['labels']
            student_segms = student_result['segms']
            # assert bboxes is not None and len(bboxes) > 0
            if topk > 0:
                teacher_idxs = np.argsort(-teacher_scores)
                student_idxs = np.argsort(-student_scores)
                teacher_bboxes = teacher_bboxes[teacher_idxs[:topk]]
                student_bboxes = student_bboxes[student_idxs[:topk]]
                teacher_labels = teacher_labels[teacher_idxs[:topk]]
                student_labels = student_labels[student_idxs[:topk]]
                if teacher_segms is not None:
                    teacher_segms = teacher_segms[teacher_idxs[:topk]]
                    student_segms = student_segms[student_idxs[:topk]]
            teacher_targets = [
                DetBoxScoreTarget(bboxes=teacher_bboxes, labels=teacher_labels, segms=teacher_segms, 
                    device = device_)
            ]
            student_targets = [
                DetBoxScoreTarget(bboxes=student_bboxes, labels=student_labels, segms=student_segms, 
                    device = device_)
            ]

            if method in GRAD_BASE_METHOD_MAP:
                self.teacher_cam_model.set_return_loss(True)
                self.teacher_cam_model.set_input_data(image, bboxes=teacher_bboxes, labels=teacher_labels)
                self.teacher_det_cam_visualizer.switch_activations_and_grads(self.teacher_cam_model)
                self.student_cam_model.set_return_loss(True)
                self.student_cam_model.set_input_data(image, bboxes=student_bboxes, labels=student_labels)
                self.student_det_cam_visualizer.switch_activations_and_grads(self.student_cam_model)

            teacher_grayscale_cam = self.teacher_det_cam_visualizer(
                image,
                targets=teacher_targets,
                aug_smooth=False,
                eigen_smooth=False) # (1, 375, 1242)

            student_grayscale_cam = self.student_det_cam_visualizer(
                image,
                targets=student_targets,
                aug_smooth=False,
                eigen_smooth=False) # (1, 375, 1242)

            # image_with_bounding_boxes = self.teacher_det_cam_visualizer.show_cam(
            #     image, teacher_bboxes, teacher_labels, teacher_grayscale_cam[0], with_norm_in_bboxes=False)
            # mmcv.mkdir_or_exist('../feature_map/teacher/')
            # out_file = os.path.join('../feature_map/teacher/', os.path.basename(image_path))
            # mmcv.imwrite(image_with_bounding_boxes, out_file)

            # image_with_bounding_boxes = self.student_det_cam_visualizer.show_cam(
            #     image, student_bboxes, student_labels, student_grayscale_cam[0], with_norm_in_bboxes=False)
            # mmcv.mkdir_or_exist('../feature_map/student/')
            # out_file = os.path.join('../feature_map/student/', os.path.basename(image_path))
            # mmcv.imwrite(image_with_bounding_boxes, out_file)


            if method in GRAD_BASE_METHOD_MAP:
                self.teacher_cam_model.set_return_loss(False)
                self.teacher_det_cam_visualizer.switch_activations_and_grads(self.teacher_cam_model)
                self.student_cam_model.set_return_loss(False)
                self.student_det_cam_visualizer.switch_activations_and_grads(self.student_cam_model)
            teacher_cam_masks.append(teacher_grayscale_cam)
            student_cam_masks.append(student_grayscale_cam)

        teacher_cam_mask_per_layer = []
        student_cam_mask_per_layer = []
        for j in range(0,4):
            teacher_cam_mask_per_batch = []
            student_cam_mask_per_batch = []
            for i in range(N):
                target_size = self.get_target_width_height(feat[j])
                new_teacher_cam_mask_per_layer = torch.from_numpy(cv2.resize(np.transpose(teacher_cam_masks[i][j], (1,2,0)), target_size))
                new_student_cam_mask_per_layer = torch.from_numpy(cv2.resize(np.transpose(student_cam_masks[i][j], (1,2,0)), target_size))
                teacher_cam_mask_per_batch.append(new_teacher_cam_mask_per_layer)
                student_cam_mask_per_batch.append(new_student_cam_mask_per_layer)
            teacher_cam_mask_per_layer.append(torch.stack(teacher_cam_mask_per_batch, 0))
            student_cam_mask_per_layer.append(torch.stack(student_cam_mask_per_batch, 0))


        self.student.train()
        student_loss = self.student.forward_train(img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, **kwargs)

        buffer_dict = dict(self.named_buffers())
        for i, item_loc in enumerate(self.distill_cfg):
            
            student_module = 'student_' + item_loc.student_module.replace('.','_')
            teacher_module = 'teacher_' + item_loc.teacher_module.replace('.','_')
            student_feat = buffer_dict[student_module]
            teacher_feat = buffer_dict[teacher_module]

            for item_loss in item_loc.methods:
                loss_name = item_loss.name
                
                student_loss[loss_name] = self.distill_losses[loss_name](student_feat,teacher_feat, teacher_cam_mask_per_layer[3-i], student_cam_mask_per_layer[3-i], gt_bboxes, img_metas)
        
        
        return student_loss

    def get_target_width_height(self,
                                input_tensor: torch.Tensor) -> Tuple[int, int]:
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height
    
    def simple_test(self, img, img_metas, **kwargs):
        return self.student.simple_test(img, img_metas, **kwargs)
    def aug_test(self, imgs, img_metas, **kwargs):
        return self.student.aug_test(imgs, img_metas, **kwargs)
    def extract_feat(self, imgs):
        """Extract features from images."""
        return self.student.extract_feat(imgs)


