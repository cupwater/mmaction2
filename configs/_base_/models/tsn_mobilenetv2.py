'''
Author: Peng Bo
Date: 2022-11-28 15:48:31
LastEditTime: 2022-11-28 15:55:27
Description: 

'''
# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='MobileNetV2',
        pretrained='mmcls://mobilenet_v2',
        norm_eval=False),
    cls_head=dict(
        type='TSNHead',
        num_classes=6,
        in_channels=512,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.4,
        init_std=0.01),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips=None))
