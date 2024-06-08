# GKD
WACV 2024 Paper: Gradient-Guided Knowledge Distillation for Object Detectors
----------------------

## Install MMDetection and grad-cam
  - Our codes are based on [MMDetection-2.x](https://mmdetection.readthedocs.io/en/v2.25.0/). Please follow the installation of MMDetection and make sure you can run it successfully.
  - This repo uses mmdet==2.25.3 and mmcv-full==1.7.0
  - install grad-cam by: ```pip install grad-cam```

## Add and Replace the codes
  - Add the configs/. in our codes to the configs/ in mmdetectin's codes.
  - Add the mmdet/distillation/. in our codes to the mmdet/ in mmdetectin's codes.
  - Replace the mmdet/apis/train.py and tools/train.py in mmdetection's codes with mmdet/apis/train.py and tools/train.py in our codes.
  - Add pth_transfer.py to mmdetection's codes.
  - Unzip COCO dataset into data/coco/, use [filter_traffic.py]() to get coco-traffic dataset.
  - Unzip KITTI dataset into data/kitti/ (We group classes as 3 and split orignal training set into 8:2 as training and validation sets)

## Train

```
#single GPU
python tools/train.py configs/distillers/gkd/gkd_faster_rcnn_r50_r101_fpn_1x_coco.py

#multi GPU
bash tools/dist_train.sh configs/distillers/gkd/gkd_faster_rcnn_r50_r101_fpn_1x_coco.py 8
```

## Transfer
```
# Tansfer the FGD model into mmdet model
python pth_transfer.py --fgd_path $fgd_ckpt --output_path $new_mmdet_ckpt
```
## Test

```
#single GPU
python tools/test.py configs/distillers/gkd/gkd_faster_rcnn_r50_r101_fpn_1x_coco.py $new_mmdet_ckpt --eval bbox

#multi GPU
bash tools/dist_test.sh configs/distillers/gkd/gkd_faster_rcnn_r50_r101_fpn_1x_coco.py $new_mmdet_ckpt 8 --eval bbox
```



## Citation
```
@inproceedings{lan2024gradient,
  title={Gradient-guided knowledge distillation for object detectors},
  author={Lan, Qizhen and Tian, Qing},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={424--433},
  year={2024}
}
```


## Acknowledgement

Our code is based on the project [MMDetection](https://github.com/open-mmlab/mmdetection).

Thanks to the work [FGD](https://github.com/yzd-v/FGD) and [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam/tree/master).
