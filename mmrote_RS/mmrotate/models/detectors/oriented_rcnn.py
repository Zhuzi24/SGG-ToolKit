# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import ROTATED_DETECTORS
from .two_stage import RotatedTwoStageDetector

from maskrcnn_benchmark.modeling.detector.roted_RS import RotatedDetector_large
from maskrcnn_benchmark.modeling.detector.two_stage_RS_multi_12 import RotatedTwoStageDetector_Mul
# from maskrcnn_benchmark.modeling.detector.roted import ROTATED

from .two_stage_small import RotatedTwoStageDetector_small
@ROTATED_DETECTORS.register_module()
class OrientedRCNN(RotatedTwoStageDetector):
    """Implementation of `Oriented R-CNN for Object Detection.`__

    __ https://openaccess.thecvf.com/content/ICCV2021/papers/Xie_Oriented_R-CNN_for_Object_Detection_ICCV_2021_paper.pdf  # noqa: E501, E261.
    """

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(OrientedRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        
    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmrotate/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 6).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs


# @ROTATED_DETECTORS.register_module()
# class OrientedRCNN(RotatedDetector_large):
#     """Implementation of `Oriented R-CNN for Object Detection.`__

#     __ https://openaccess.thecvf.com/content/ICCV2021/papers/Xie_Oriented_R-CNN_for_Object_Detection_ICCV_2021_paper.pdf  # noqa: E501, E261.
#     """

#     def __init__(self,
#                  backbone,
#                  rpn_head,
#                  roi_head,
#                  train_cfg,
#                  test_cfg,
#                  neck=None,
#                  pretrained=None,
#                  init_cfg=None):
#         super(OrientedRCNN, self).__init__(
#             backbone=backbone,
#             neck=neck,
#             rpn_head=rpn_head,
#             roi_head=roi_head,
#             train_cfg=train_cfg,
#             test_cfg=test_cfg,
#             pretrained=pretrained,
#             init_cfg=init_cfg)

#     def forward_dummy(self, img):
#         """Used for computing network flops.

#         See `mmrotate/tools/analysis_tools/get_flops.py`
#         """
#         outs = ()
#         # backbone
#         x = self.extract_feat(img)
#         # rpn
#         if self.with_rpn:
#             rpn_outs = self.rpn_head(x)
#             outs = outs + (rpn_outs, )
#         proposals = torch.randn(1000, 6).to(img.device)
#         # roi_head
#         roi_outs = self.roi_head.forward_dummy(x, proposals)
#         outs = outs + (roi_outs, )
#         return outs