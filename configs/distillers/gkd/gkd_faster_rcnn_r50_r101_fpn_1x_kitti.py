_base_ = [
    '../../_base_/datasets/kitti_detection.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]
# model settings
find_unused_parameters=True
# temp=0.5
# alpha_fgd=0.001
# beta_fgd=0
# gamma_fgd=0.0005
# lambda_fgd=0.000005

temp=0.5
alpha_fgd=0
beta_fgd=0.5 #0.05
gamma_fgd=0
lambda_fgd=0
distiller = dict(
    type='Two_Stage_GKD_DetectionDistiller',
    teacher_pretrained = '/users/PCS0252/lanqz7766/mmdetection/KD_kitti_results/faster_rcnn_r101_fpn_2x_kitti/epoch_24.pth',
    init_student = True,
    distill_cfg = [ dict(student_module = 'neck.fpn_convs.3.conv',
                         teacher_module = 'neck.fpn_convs.3.conv',
                         output_hook = True,
                         methods=[dict(type='GKD_FeatureLoss',
                                       name='loss_fgd_fpn_3',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fgd=alpha_fgd,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.2.conv',
                         teacher_module = 'neck.fpn_convs.2.conv',
                         output_hook = True,
                         methods=[dict(type='GKD_FeatureLoss',
                                       name='loss_fgd_fpn_2',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fgd=alpha_fgd,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.1.conv',
                         teacher_module = 'neck.fpn_convs.1.conv',
                         output_hook = True,
                         methods=[dict(type='GKD_FeatureLoss',
                                       name='loss_fgd_fpn_1',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fgd=alpha_fgd,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       )
                                ]
                        ),
                    dict(student_module = 'neck.fpn_convs.0.conv',
                         teacher_module = 'neck.fpn_convs.0.conv',
                         output_hook = True,
                         methods=[dict(type='GKD_FeatureLoss',
                                       name='loss_fgd_fpn_0',
                                       student_channels = 256,
                                       teacher_channels = 256,
                                       temp = temp,
                                       alpha_fgd=alpha_fgd,
                                       beta_fgd=beta_fgd,
                                       gamma_fgd=gamma_fgd,
                                       lambda_fgd=lambda_fgd,
                                       )
                                ]
                        ),

                   ]
    )

student_cfg = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_kitti.py'
teacher_cfg = 'configs/faster_rcnn/faster_rcnn_r101_fpn_2x_kitti.py'
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
checkpoint_config = dict(interval=6)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

