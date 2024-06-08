_base_ = './faster_rcnn_r50_fpn_2x_kitti.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
