# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import copy
import cv2
import mmcv
import numpy as np
import torch
import torch.nn as nn
import torchvision
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
import matplotlib.pyplot as plt
# from mmdet.utils.basecam import BaseCAM
try:
    from pytorch_grad_cam import (AblationCAM, AblationLayer,
                                  ActivationsAndGradients)
    # from pytorch_grad_cam.base_cam import BaseCAM
    # from pytorch_grad_cam.utils.image import scale_cam_image, show_cam_on_image
    from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
except ImportError:
    raise ImportError('Please run `pip install "grad-cam"` to install '
                      '3rd party package pytorch_grad_cam.')

from mmdet.core import get_classes
# from mmdet.datasets import replace_ImageToTensor
import warnings
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
import ttach as tta
from typing import Callable, List, Tuple
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from mmdet.utils import get_root_logger
logger = get_root_logger()



def replace_ImageToTensor(pipelines):
    """Replace the ImageToTensor transform in a data pipeline to
    DefaultFormatBundle, which is normally useful in batch inference.

    Args:
        pipelines (list[dict]): Data pipeline configs.

    Returns:
        list: The new pipeline list with all ImageToTensor replaced by
            DefaultFormatBundle.
    """
    pipelines = copy.deepcopy(pipelines)
    for i, pipeline in enumerate(pipelines):
        if pipeline['type'] == 'MultiScaleFlipAug':
            assert 'transforms' in pipeline
            pipeline['transforms'] = replace_ImageToTensor(
                pipeline['transforms'])
        elif pipeline['type'] == 'ImageToTensor':
            warnings.warn(
                '"ImageToTensor" pipeline is replaced by '
                '"DefaultFormatBundle" for batch inference. It is '
                'recommended to manually replace it in the test '
                'data pipeline in your config file.', UserWarning)
            pipelines[i] = {'type': 'DefaultFormatBundle'}
    return pipelines

def scale_cam_image_hw(cam, target_size=None):
    result = []
    for img in cam:
        if target_size is not None:
            img = cv2.resize(img, target_size)
        result.append(img)
        # except ValueError:  #raised if 'img' is empty
        #     pass

    result = np.float32(result)

    return result

def scale_cam_image(cam, target_size=None):
    result = []
    for img in cam:
        try:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        except ValueError:  #raised if 'img' is empty
            pass

    result = np.float32(result)

    return result



class BaseCAM:
    def __init__(self,
                 model: torch.nn.Module,
                 target_layers: List[torch.nn.Module],
                 use_cuda: bool = False,
                 reshape_transform: Callable = None,
                 compute_input_gradient: bool = False,
                 uses_gradients: bool = True) -> None:
        self.model = model.eval()
        self.target_layers = target_layers
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    def get_cam_weights(self,
                        input_tensor: torch.Tensor,
                        target_layers: List[torch.nn.Module],
                        targets: List[torch.nn.Module],
                        activations: torch.Tensor,
                        grads: torch.Tensor) -> np.ndarray:
        raise Exception("Not Implemented")
        # return np.mean(grads, axis=(2, 3))

    def get_cam_image(self,
                      input_tensor: torch.Tensor,
                      target_layer: torch.nn.Module,
                      targets: List[torch.nn.Module],
                      activations: torch.Tensor,
                      grads: torch.Tensor,
                      eigen_smooth: bool = False) -> np.ndarray:

        weights = self.get_cam_weights(input_tensor,
                                       target_layer,
                                       targets,
                                       activations,
                                       grads) #(1,256)
        # logger.info(f'activation number of negative values: {np.sum(activations > 0)}')

        weighted_activations = weights[:, :, None, None] * activations ## (1, 256, 20, 20)
        # logger.info(f"weighted cam weights shape is: {weighted_activations.shape}")
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1) ##  (1, 20, 20)

            # cam = weighted_activations ## (1, 256, 20,20 )
        return cam

    def forward(self,
                input_tensor: torch.Tensor,
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False) -> np.ndarray:

        if self.cuda:
            input_tensor = input_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)

        outputs = self.activations_and_grads(input_tensor)
        # logger.info(f"outputs shape is: {outputs[0]}")
        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(
                category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output)
                       for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   targets,
                                                   eigen_smooth)
        # return self.aggregate_multi_layers(cam_per_layer) # (1, 375, 1242) ##for visulization
        return cam_per_layer # (1, 1, 375, 1242) ###for distilaltion

    def get_target_width_height(self,
                                input_tensor: torch.Tensor) -> Tuple[int, int]:
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(
            self,
            input_tensor: torch.Tensor,
            targets: List[torch.nn.Module],
            eigen_smooth: bool) -> np.ndarray:
        # input_tensor = torch.from_numpy(input_tensor).permute(2, 0, 1)
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i] # (1, 256, 20, 20)

            cam = self.get_cam_image(input_tensor,
                                     target_layer,
                                     targets,
                                     layer_activations,
                                     layer_grads,
                                     eigen_smooth)  # (1, 20 ,20)

            cam = np.maximum(cam, 0) # (1, 20, 20) #### remove relu for testing

            scaled = scale_cam_image(cam, target_size=target_size) ##
            # cam_per_target_layer.append(scaled[:, None, :]) ## For vis

            cam_per_target_layer.append(scaled) ## for distillationl, comments out if while vis
        cam_per_target_layer = np.asarray(cam_per_target_layer) ## for distillationl, comments out if while vis
        return cam_per_target_layer # (1, 1, w, h)

    def aggregate_multi_layers(
            self,
            cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        # return scale_cam_image(result) ##orignal version
        return scale_cam_image(result)

    def forward_augmentation_smoothing(self,
                                       input_tensor: torch.Tensor,
                                       targets: List[torch.nn.Module],
                                       eigen_smooth: bool = False) -> np.ndarray:
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,
                               targets,
                               eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(self,
                 input_tensor: torch.Tensor,
                 targets: List[torch.nn.Module] = None,
                 aug_smooth: bool = False,
                 eigen_smooth: bool = False) -> np.ndarray:

        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(
                input_tensor, targets, eigen_smooth)

        return self.forward(input_tensor,
                            targets, eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True



 ############################           
def reshape_transform(feats, max_shape=(-1, -1), is_need_grad=False):
    """Reshape and aggregate feature maps when the input is a multi-layer
    feature map.

    Takes these tensors with different sizes, resizes them to a common shape,
    and concatenates them.
    """
    if len(max_shape) == 1:
        max_shape = max_shape * 2

    if isinstance(feats, torch.Tensor):
        feats = [feats]
    else:
        if is_need_grad:
            raise NotImplementedError('The `grad_base` method does not '
                                      'support output multi-activation layers')

    max_h = max([im.shape[-2] for im in feats])
    max_w = max([im.shape[-1] for im in feats])
    if -1 in max_shape:
        max_shape = (max_h, max_w)
    else:
        max_shape = (min(max_h, max_shape[0]), min(max_w, max_shape[1]))

    activations = []
    for feat in feats:
        activations.append(
            torch.nn.functional.interpolate(
                torch.abs(feat), max_shape, mode='bilinear'))

    activations = torch.cat(activations, axis=1)
    return activations

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


class DetCAMModel(nn.Module):
    """Wrap the mmdet model class to facilitate handling of non-tensor
    situations during inference."""

    def __init__(self, cfg, checkpoint, score_thr, device='cuda:0'):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.score_thr = score_thr
        self.checkpoint = checkpoint
        self.detector = self.build_detector()

        self.return_loss = False
        self.input_data = None
        self.img = None

    def build_detector(self):
        cfg = copy.deepcopy(self.cfg)

        detector = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        ###################
        if 'custom_hooks' in cfg:
            for hook in cfg.custom_hooks:
                if hook.type == 'FisherPruningHook' or hook.type == "PruningHook":
                    hook_cfg = hook.copy()
                    hook_cfg.pop('priority', None)
                    from mmcv.runner.hooks import HOOKS
                    hook_cls = HOOKS.get(hook_cfg['type'])
                    if hasattr(hook_cls, 'after_build_model'):
                        pruning_hook = mmcv.build_from_cfg(hook_cfg, HOOKS)
                        pruning_hook.after_build_model(detector)
        ###################
        if self.checkpoint is not None:
            checkpoint = load_checkpoint(
                detector, self.checkpoint, map_location='cpu')
            if 'CLASSES' in checkpoint.get('meta', {}):
                detector.CLASSES = checkpoint['meta']['CLASSES']
            else:
                import warnings
                warnings.simplefilter('once')
                warnings.warn('Class names are not saved in the checkpoint\'s '
                              'meta data, use COCO classes by default.')
                detector.CLASSES = get_classes('coco')

        detector.to(self.device)
        detector.eval()
        return detector

    def set_return_loss(self, return_loss):
        self.return_loss = return_loss

    def update_model_weights(self, state_dict):
        self.detector.load_state_dict(state_dict)

    def set_input_data(self, img, bboxes=None, labels=None):
        self.img = img
        cfg = copy.deepcopy(self.cfg)
        if self.return_loss:
            assert bboxes is not None
            assert labels is not None
            cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
            cfg.data.test.pipeline[1].transforms[-1] = dict(
                type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            test_pipeline = Compose(cfg.data.test.pipeline)
            # TODO: support mask
            data = dict(
                img=self.img,
                gt_bboxes=bboxes,
                gt_labels=labels.astype(np.long),
                bbox_fields=['gt_bboxes'])
            data = test_pipeline(data)
            data = collate([data], samples_per_gpu=1)

            # just get the actual data from DataContainer
            data['img_metas'] = [
                img_metas.data[0][0] for img_metas in data['img_metas']
            ]
            data['img'] = [img.data[0] for img in data['img']]
            data['gt_bboxes'] = [
                gt_bboxes.data[0] for gt_bboxes in data['gt_bboxes']
            ]
            data['gt_labels'] = [
                gt_labels.data[0] for gt_labels in data['gt_labels']
            ]
            if next(self.detector.parameters()).is_cuda:
                # scatter to specified GPU
                data = scatter(data, [self.device])[0]
            data['img'] = data['img'][0]
            data['gt_bboxes'] = data['gt_bboxes'][0]
            data['gt_labels'] = data['gt_labels'][0]
        else:
            # set loading pipeline type
            cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
            data = dict(img=self.img)
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
            test_pipeline = Compose(cfg.data.test.pipeline)
            data = test_pipeline(data)
            data = collate([data], samples_per_gpu=1)
            # just get the actual data from DataContainer
            data['img_metas'] = [
                img_metas.data[0] for img_metas in data['img_metas']
            ]
            data['img'] = [img.data[0] for img in data['img']]

            if next(self.detector.parameters()).is_cuda:
                # scatter to specified GPU
                data = scatter(data, [self.device])[0]
            else:
                for m in self.detector.modules():
                    assert not isinstance(
                        m, RoIPool
                    ), 'CPU inference with RoIPool is not supported currently.'

        self.input_data = data

    def __call__(self, *args, **kwargs):
        assert self.input_data is not None
        if self.return_loss:
            loss = self.detector(return_loss=True, **self.input_data)
            return [loss]
        else:
            with torch.no_grad():
                results = self.detector(
                    return_loss=False, rescale=True, **self.input_data)[0]

                if isinstance(results, tuple):
                    bbox_result, segm_result = results
                    if isinstance(segm_result, tuple):
                        segm_result = segm_result[0]  # ms rcnn
                else:
                    bbox_result, segm_result = results, None

                bboxes = np.vstack(bbox_result)
                labels = [
                    np.full(bbox.shape[0], i, dtype=np.int32)
                    for i, bbox in enumerate(bbox_result)
                ]
                labels = np.concatenate(labels)

                segms = None
                if segm_result is not None and len(labels) > 0:  # non empty
                    segms = mmcv.concat_list(segm_result)
                    if isinstance(segms[0], torch.Tensor):
                        segms = torch.stack(
                            segms, dim=0).detach().cpu().numpy()
                    else:
                        segms = np.stack(segms, axis=0)

                if self.score_thr > 0:
                    assert bboxes is not None and bboxes.shape[1] == 5
                    scores = bboxes[:, -1]
                    inds = scores > self.score_thr
                    bboxes = bboxes[inds, :]
                    labels = labels[inds]
                    if segms is not None:
                        segms = segms[inds, ...]
                return [{'bboxes': bboxes, 'labels': labels, 'segms': segms}]


class DetAblationLayer(AblationLayer):

    def __init__(self):
        super(DetAblationLayer, self).__init__()
        self.activations = None

    def set_next_batch(self, input_batch_index, activations,
                       num_channels_to_ablate):
        """Extract the next batch member from activations, and repeat it
        num_channels_to_ablate times."""
        if isinstance(activations, torch.Tensor):
            return super(DetAblationLayer,
                         self).set_next_batch(input_batch_index, activations,
                                              num_channels_to_ablate)

        self.activations = []
        for activation in activations:
            activation = activation[
                input_batch_index, :, :, :].clone().unsqueeze(0)
            self.activations.append(
                activation.repeat(num_channels_to_ablate, 1, 1, 1))

    def __call__(self, x):
        """Go over the activation indices to be ablated, stored in
        self.indices.

        Map between every activation index to the tensor in the Ordered Dict
        from the FPN layer.
        """
        result = self.activations

        if isinstance(result, torch.Tensor):
            return super(DetAblationLayer, self).__call__(x)

        channel_cumsum = np.cumsum([r.shape[1] for r in result])
        num_channels_to_ablate = result[0].size(0)  # batch
        for i in range(num_channels_to_ablate):
            pyramid_layer = bisect.bisect_right(channel_cumsum,
                                                self.indices[i])
            if pyramid_layer > 0:
                index_in_pyramid_layer = self.indices[i] - channel_cumsum[
                    pyramid_layer - 1]
            else:
                index_in_pyramid_layer = self.indices[i]
            result[pyramid_layer][i, index_in_pyramid_layer, :, :] = -1000
        return result


class DetCAMVisualizer:
    """mmdet cam visualization class.

    Args:
        method:  CAM method. Currently supports
           `ablationcam`,`eigencam` and `featmapam`.
        model (nn.Module): MMDet model.
        target_layers (list[torch.nn.Module]): The target layers
            you want to visualize.
        reshape_transform (Callable, optional): Function of Reshape
            and aggregate feature maps. Defaults to None.
    """

    def __init__(self,
                 method_class,
                 model,
                 target_layers,
                 reshape_transform=None,
                 is_need_grad=False,
                 extra_params=None):
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.is_need_grad = is_need_grad

        if method_class.__name__ == 'AblationCAM':
            batch_size = extra_params.get('batch_size', 1)
            ratio_channels_to_ablate = extra_params.get(
                'ratio_channels_to_ablate', 1.)
            self.cam = AblationCAM(
                model,
                target_layers,
                use_cuda=True if 'cuda' in model.device else False,
                reshape_transform=reshape_transform,
                batch_size=batch_size,
                ablation_layer=extra_params['ablation_layer'],
                ratio_channels_to_ablate=ratio_channels_to_ablate)
        else:
            self.cam = method_class(
                model,
                target_layers,
                use_cuda=True if 'cuda' in model.device else False,
                reshape_transform=reshape_transform,
            )
            if self.is_need_grad:
                self.cam.activations_and_grads.release()

        self.classes = model.detector.CLASSES
        # self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3)) ## changed
        num_classes = 80

        # Generate a list of distinct colors
        colors_ = plt.cm.jet(range(num_classes))

        # Convert colors to RGB format
        self.COLORS = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in colors_]

        # self.COLORS = [(255, 0, 0),  # Red
        #        (0, 255, 0),  # Green
        #        (0, 0, 255)]  # Blue

    def switch_activations_and_grads(self, model):
        self.cam.model = model

        if self.is_need_grad is True:
            self.cam.activations_and_grads = ActivationsAndGradients(
                model, self.target_layers, self.reshape_transform)
            self.is_need_grad = False
        else:
            self.cam.activations_and_grads.release()
            self.is_need_grad = True

    def __call__(self, img, targets, aug_smooth=False, eigen_smooth=False):
        img = torch.from_numpy(img)[None].permute(0, 3, 1, 2)
        # logger.info(f"img shape is {img.shape}")
        # gray_cam = self.cam(img, targets, aug_smooth, eigen_smooth)
        return self.cam(img, targets, aug_smooth, eigen_smooth)

    def get_target_width_height(self,
                                input_tensor: torch.Tensor) -> Tuple[int, int]:
        width, height = input_tensor.shape[1], input_tensor.shape[0]
        return width, height

    def show_cam(self,
                 image,
                 boxes,
                 labels,
                 grayscale_cam,
                 with_norm_in_bboxes=False):
        """Normalize the CAM to be in the range [0, 1] inside every bounding
        boxes, and zero outside of the bounding boxes."""
        # image = torch.from_numpy(image)[None].permute(0, 3, 1, 2)
        target_size = self.get_target_width_height(image)

        # grayscale_cam = np.transpose(scale_cam_image(grayscale_cam, target_size), (1, 2, 0)) #### moved for bathc norm
        grayscale_cam = np.transpose(scale_cam_image(grayscale_cam, target_size), (1, 2, 0))
        if (with_norm_in_bboxes is True) and (len(boxes)!= 0):
            boxes = boxes.astype(np.int32)
            renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
            images = []
            for x1, y1, x2, y2 in boxes:
                img = renormalized_cam * 0
                img[y1:y2,
                    x1:x2] = scale_cam_image(grayscale_cam[y1:y2,
                                                           x1:x2].copy())
                images.append(img)
            # logger.info(f'renormalized_camshape is: {renormalized_cam.shape}')
            # logger.info(f'images length is: {len(images)}')
            # logger.info(f'images shape is: {images[0].shape}')
            renormalized_cam = np.max(np.float32(images), axis = 0) #
            # logger.info(f'renormalized_camshape is: {renormalized_cam.shape}')
            renormalized_cam = scale_cam_image(renormalized_cam)
        else:
            renormalized_cam = grayscale_cam

        cam_image_renormalized = show_cam_on_image(
            image / 255, renormalized_cam, use_rgb=False)

        # image_with_bounding_boxes = self._draw_boxes(boxes, labels,
        #                                              cam_image_renormalized)
        # return image_with_bounding_boxes
        return cam_image_renormalized

    def _draw_boxes(self, boxes, labels, image):
        for i, box in enumerate(boxes):
            label = labels[i]
            color = self.COLORS[label]
            cv2.rectangle(image, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), color, 2)
            cv2.putText(
                image,
                self.classes[label], (int(box[0]), int(box[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                lineType=cv2.LINE_AA)
        return image

class DetBoxScoreTarget:
    """For every original detected bounding box specified in "bboxes",
    assign a score on how the current bounding boxes match it,
        1. In Bbox IoU
        2. In the classification score.
        3. In Mask IoU if ``segms`` exist.

    If there is not a large enough overlap, or the category changed,
    assign a score of 0.

    The total score is the sum of all the box scores.
    """

    def __init__(self,
                 bboxes,
                 labels,
                 segms=None,
                 match_iou_thr=0.5,
                 device='cuda:0'):
        assert len(bboxes) == len(labels)
        self.focal_bboxes = torch.from_numpy(bboxes).to(device=device)
        self.focal_labels = labels
        if segms is not None:
            assert len(bboxes) == len(segms)
            self.focal_segms = torch.from_numpy(segms).to(device=device)
        else:
            self.focal_segms = [None] * len(labels)
        self.match_iou_thr = match_iou_thr

        self.device = device

    def __call__(self, results):
        output = torch.tensor([0.], device=self.device)

        if 'loss_cls' in results:
            # grad_base_method
            for loss_key, loss_value in results.items():
                if 'loss' not in loss_key:
                    continue
                if isinstance(loss_value, list):
                    output += sum(loss_value)
                else:
                    output += loss_value
            return output
        else:
            # grad_free_method
            if len(results['bboxes']) == 0:
                return output

            pred_bboxes = torch.from_numpy(results['bboxes']).to(self.device)
            pred_labels = results['labels']
            pred_segms = results['segms']

            if pred_segms is not None:
                pred_segms = torch.from_numpy(pred_segms).to(self.device)

            for focal_box, focal_label, focal_segm in zip(
                    self.focal_bboxes, self.focal_labels, self.focal_segms):
                ious = torchvision.ops.box_iou(focal_box[None],
                                               pred_bboxes[..., :4])
                index = ious.argmax()
                if ious[0, index] > self.match_iou_thr and pred_labels[
                        index] == focal_label:
                    # TODO: Adaptive adjustment of weights based on algorithms
                    score = ious[0, index] + pred_bboxes[..., 4][index]
                    output = output + score

                    if focal_segm is not None and pred_segms is not None:
                        segms_score = (focal_segm *
                                       pred_segms[index]).sum() / (
                                           focal_segm.sum() +
                                           pred_segms[index].sum() + 1e-7)
                        output = output + segms_score
            return output


# TODO: Fix RuntimeError: element 0 of tensors does not require grad and
#  does not have a grad_fn.
#  Can be removed once the source code is fixed.
class EigenCAM(BaseCAM):

    def __init__(self,
                 model,
                 target_layers,
                 use_cuda=False,
                 reshape_transform=None):
        super(EigenCAM, self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform,
            uses_gradients=False)

    def get_cam_image(self, input_tensor, target_layer, target_category,
                      activations, grads, eigen_smooth):
        return get_2d_projection(activations)


class FeatmapAM(EigenCAM):
    """Visualize Feature Maps.

    Visualize the (B,C,H,W) feature map averaged over the channel dimension.
    """

    def __init__(self,
                 model,
                 target_layers,
                 use_cuda=False,
                 reshape_transform=None):
        super(FeatmapAM, self).__init__(model, target_layers, use_cuda,
                                        reshape_transform)

    def get_cam_image(self, input_tensor, target_layer, target_category,
                      activations, grads, eigen_smooth):
        return np.mean(activations, axis=1)

class GradCAM(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(
            GradCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        # logger.info(f'grads have neg value number is :{np.sum(grads <= 0)}')
        return np.mean(grads, axis=(2, 3))


class GradCAMPlusPlus(BaseCAM):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(GradCAMPlusPlus, self).__init__(model, target_layers, use_cuda,
                                              reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layers,
                        target_category,
                        activations,
                        grads):
        grads_power_2 = grads**2
        grads_power_3 = grads_power_2 * grads
        # Equation 19 in https://arxiv.org/abs/1710.11063
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 +
                               sum_activations[:, :, None, None] * grads_power_3 + eps)
        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0) * aij
        weights = np.sum(weights, axis=(2, 3))
        return weights


class LayerCAM(BaseCAM):
    def __init__(
            self,
            model,
            target_layers,
            use_cuda=False,
            reshape_transform=None):
        super(
            LayerCAM,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform)

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        # spatial_weighted_activations = np.maximum(grads, 0) * activations
        spatial_weighted_activations = np.abs(grads) * activations

        if eigen_smooth:
            cam = get_2d_projection(spatial_weighted_activations)
        else:
            cam = spatial_weighted_activations.sum(axis=1)
        return cam
        
class Student_DetCAMModel(nn.Module):
    """Wrap the mmdet model class to facilitate handling of non-tensor
    situations during inference."""

    def __init__(self, cfg, checkpoint, detector, score_thr, device='cuda:0'):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.score_thr = score_thr
        self.checkpoint = checkpoint
        self.detector = detector.eval()
        self.teacher_detector = self.build_detector()

        cfg = copy.deepcopy(self.cfg)
        teacher_detector = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))

        if self.checkpoint is not None:
            checkpoint = load_checkpoint(
                teacher_detector, self.checkpoint, map_location='cpu')
            if 'CLASSES' in checkpoint.get('meta', {}):
                detector.CLASSES = checkpoint['meta']['CLASSES']
        self.return_loss = False
        self.input_data = None
        self.img = None

    def build_detector(self):
        cfg = copy.deepcopy(self.cfg)

        detector = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))

        if self.checkpoint is not None:
            checkpoint = load_checkpoint(
                detector, self.checkpoint, map_location='cpu')
            if 'CLASSES' in checkpoint.get('meta', {}):
                detector.CLASSES = checkpoint['meta']['CLASSES']
            else:
                import warnings
                warnings.simplefilter('once')
                warnings.warn('Class names are not saved in the checkpoint\'s '
                              'meta data, use COCO classes by default.')
                detector.CLASSES = get_classes('coco')

        detector.to(self.device)
        detector.eval()
        return detector

    def set_return_loss(self, return_loss):
        self.return_loss = return_loss

    def update_model_weights(self, state_dict):
        self.detector.load_state_dict(state_dict)

    def set_input_data(self, img, bboxes=None, labels=None):
        self.img = img
        cfg = copy.deepcopy(self.cfg)
        if self.return_loss:
            assert bboxes is not None
            assert labels is not None
            cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
            cfg.data.test.pipeline[1].transforms[-1] = dict(
                type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            test_pipeline = Compose(cfg.data.test.pipeline)
            # TODO: support mask
            data = dict(
                img=self.img,
                gt_bboxes=bboxes,
                gt_labels=labels.astype(np.long),
                bbox_fields=['gt_bboxes'])
            data = test_pipeline(data)
            data = collate([data], samples_per_gpu=1)

            # just get the actual data from DataContainer
            data['img_metas'] = [
                img_metas.data[0][0] for img_metas in data['img_metas']
            ]
            data['img'] = [img.data[0] for img in data['img']]
            data['gt_bboxes'] = [
                gt_bboxes.data[0] for gt_bboxes in data['gt_bboxes']
            ]
            data['gt_labels'] = [
                gt_labels.data[0] for gt_labels in data['gt_labels']
            ]
            if next(self.detector.parameters()).is_cuda:
                # scatter to specified GPU
                data = scatter(data, [self.device])[0]

            data['img'] = data['img'][0]
            data['gt_bboxes'] = data['gt_bboxes'][0]
            data['gt_labels'] = data['gt_labels'][0]
        else:
            # set loading pipeline type
            cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
            data = dict(img=self.img)
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
            test_pipeline = Compose(cfg.data.test.pipeline)
            data = test_pipeline(data)
            data = collate([data], samples_per_gpu=1)
            # just get the actual data from DataContainer
            data['img_metas'] = [
                img_metas.data[0] for img_metas in data['img_metas']
            ]
            data['img'] = [img.data[0] for img in data['img']]

            if next(self.detector.parameters()).is_cuda:
                # scatter to specified GPU
                data = scatter(data, [self.device])[0]
            else:
                for m in self.detector.modules():
                    assert not isinstance(
                        m, RoIPool
                    ), 'CPU inference with RoIPool is not supported currently.'

        self.input_data = data

    def __call__(self, *args, **kwargs):
        assert self.input_data is not None
        if self.return_loss:
            loss = self.detector(return_loss=True, **self.input_data)
            return [loss]
        else:
            with torch.no_grad():
                results = self.detector(
                    return_loss=False, rescale=True, **self.input_data)[0]

                if isinstance(results, tuple):
                    bbox_result, segm_result = results
                    if isinstance(segm_result, tuple):
                        segm_result = segm_result[0]  # ms rcnn
                else:
                    bbox_result, segm_result = results, None

                bboxes = np.vstack(bbox_result)
                labels = [
                    np.full(bbox.shape[0], i, dtype=np.int32)
                    for i, bbox in enumerate(bbox_result)
                ]
                labels = np.concatenate(labels)

                segms = None
                if segm_result is not None and len(labels) > 0:  # non empty
                    segms = mmcv.concat_list(segm_result)
                    if isinstance(segms[0], torch.Tensor):
                        segms = torch.stack(
                            segms, dim=0).detach().cpu().numpy()
                    else:
                        segms = np.stack(segms, axis=0)

                if self.score_thr > 0:
                    assert bboxes is not None and bboxes.shape[1] == 5
                    scores = bboxes[:, -1]
                    inds = scores > self.score_thr
                    bboxes = bboxes[inds, :]
                    labels = labels[inds]
                    if segms is not None:
                        segms = segms[inds, ...]
                return [{'bboxes': bboxes, 'labels': labels, 'segms': segms}]


