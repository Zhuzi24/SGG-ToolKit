# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
import numpy as np 
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_union, boxlist_intersection

import copy
import torch.nn.functional as F

from mmrotate.models.builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck

class RSROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, RS_conf):
        super(RSROIBoxHead, self).__init__()
        roi_head = RS_conf[0]
        train_cfg=RS_conf[1]
        test_cfg=RS_conf[2]
        pretrained=RS_conf[3]
        rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
        roi_head.update(train_cfg=rcnn_train_cfg)
        roi_head.update(test_cfg=test_cfg.rcnn)
        roi_head.pretrained = pretrained
        self.roi_head = build_head(roi_head)
    

    def forward(self, features = None, proposals=None, targets=None,RS_data=None):


        roi_losses = self.roi_head.forward_train(RS_data[0], RS_data[1], RS_data[2],
                                                 RS_data[3], RS_data[4],
                                                 RS_data[5], RS_data[6],
                                                 )
        return roi_losses

def build_roi_rs_box_head(RS_conf):

    return RSROIBoxHead(RS_conf)
