# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import copy
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from ..attribute_head.roi_attribute_feature_extractors import make_roi_attribute_feature_extractor
from ..box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_relation_feature_extractors import make_roi_relation_feature_extractor
from .roi_relation_predictors import make_roi_relation_predictor
from .inference import make_roi_relation_post_processor


from .loss import make_roi_relation_loss_evaluator
from .sampling import make_roi_relation_samp_processor
from maskrcnn_benchmark.modeling.roi_heads.relation_head.sema_filter import sema_sx
from maskrcnn_benchmark.modeling.roi_heads.relation_head.PPG import PPG
from maskrcnn_benchmark.modeling.roi_heads.relation_head.PPG_HBB import PPG_HBB

import json

from PIL import ImageFile 
import scipy.io as sio
import random
import math
import numpy as np
from maskrcnn_benchmark.config import cfg
import scipy.io as sio

from .utils_motifs import  to_onehot



def bbox2roi_HBB(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 5))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois

# from mmrotate.core.bbox.transforms import rbbox2roi
Tensor = torch.Tensor
# Method_flag = "OBB" # "OBB"
class ROIRelationHead(torch.nn.Module):
    """
    Generic Relation Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIRelationHead, self).__init__()
        self.cfg = cfg.clone()
        # same structure with box head, but different parameters
        # these param will be trained in a slow learning rate, while the parameters of box head will be fixed
        # Note: there is another such extractor in uniton_feature_extractor

        
        self.union_feature_extractor = make_roi_relation_feature_extractor(cfg, in_channels)
        
        if cfg.MODEL.ATTRIBUTE_ON:
            self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, half_out=True)
            self.att_feature_extractor = make_roi_attribute_feature_extractor(cfg, in_channels, half_out=True)
            feat_dim = self.box_feature_extractor.out_channels * 2
        else:
            self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
            feat_dim = self.box_feature_extractor.out_channels
        self.predictor = make_roi_relation_predictor(cfg, feat_dim)
        self.post_processor = make_roi_relation_post_processor(cfg)


        self.loss_evaluator = make_roi_relation_loss_evaluator(cfg)
        
      #  self.loss_evaluator = make_roi_relation_loss_evaluator(cfg)
        self.samp_processor = make_roi_relation_samp_processor(cfg)
        self.type = self.cfg.Type

        
        self.PPG = PPG()

        self.PPG_HBB = PPG_HBB()


    
        if "OBB" in self.type:
            from mmrotate.core.bbox.transforms import rbbox2roi
            self.rbbox2roi = rbbox2roi
        elif "HBB" in self.type:
            # from mmdet.core.bbox.transforms import rbbox2roi
            self.rbbox2roi = bbox2roi_HBB    
        self.flag =1 



        self.use_union_box = self.cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION

        self.pre = cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX and self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.tasks = "Predcls"
        elif self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX and not self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.tasks = "Sgcls"
        elif not self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX and not self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.tasks = "Sgdets"

        if cfg.Type != "CV":
            self.semafilter = sema_sx(flag=True if self.tasks == "Sgdets" else False) 


        self.filter_method = cfg.filter_method

        self.RS_Leap = cfg.RS_Leap

        self.Sema_F = cfg.Sema_F

        self.cfg = cfg
        


        self.cur_bg_c = 0.0
        self.count_dict = {}
        self.num_obj_cls = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        # parameters
        self.use_union_box = self.cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.tail_features_dict = {}
        self.cfa_features_dict = {}
        self.origial_features_dict = {}
        self.iii = 0





    def generate_random_numbers(self,N, count):
        if count > N:
            raise ValueError("Count should be less than or equal to N.")
        
        return random.sample(range(N), count)


    def generate_random_numbers(self,N, count):
        if count > N:
            raise ValueError("Count should be less than or equal to N.")        
        return random.sample(range(N), count) 

    def split_list(self,lst, max_length=350):
            return [(i, min(i + max_length, len(lst))) for i in range(0, len(lst), max_length)]

    def find_sublist_position(self, main_list, sublist):
        for i in range(len(main_list) - len(sublist) + 1):
            if sublist == main_list[i:i + len(sublist)]:
                return i  # 返回子列表的起始位置
        return 1  # 如果不存在，返回 1


    def contains_sublist(self,main_list, sublist):
        return any(sublist == main_list[i:i + len(sublist)] for i in range(len(main_list) - len(sublist) + 1))


    def generate_random_numbers(self,N, count):
        if count > N:
            raise ValueError("Count should be less than or equal to N.")
        
        return random.sample(range(N), count)
   
   
    def forward(self, features, proposals, targets=None, logger=None, ite=None, m=None, val=None,
    confu_wei =None,CCM = None,GLO_f = None,cls_new = None,OBj= None,s_f = None,MUL =None,vae = None, bce = None):

        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes. Note: it has been post-processed (regression, nms) in sgdet mode
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        
        if self.training : 
            # relation subsamples and assign ground truth label during training
            with torch.no_grad():
                if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
 
                    proposals, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.gtbox_relsample(proposals, targets)

                else:

                    proposals, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.detect_relsample(proposals, targets)
                    
        else:
            rel_labels, rel_binarys = None, None


            rel_pair_idxs = self.samp_processor.prepare_test_pairs(features[0].device, proposals)
            
            
            if self.Sema_F :
                sema_tmp = self.semafilter.sx(rel_pair_idxs=rel_pair_idxs[0], obj=proposals[0].extra_fields["labels"])
                rel_pair_idxs = sema_tmp

            
            if len(rel_pair_idxs[0]) > 10000 :

                if self.filter_method == "random_filter":
                    if not self.training:
                        print("random_filter")
                        id_num = torch.randperm(rel_pair_idxs[0].size(0))
                        rel_pair_idxs = [rel_pair_idxs[0][id_num[:10000]]]

                elif self.filter_method == "PPG":

                    print('num_PPG_before:',len(rel_pair_idxs[0]))  
                    if not self.training :
                        if "OBB" in self.type:
                            rel_pair_idxs=self.PPG.sx_Oriented(rel_pair_idxs[0],proposals)
                        else:
                            rel_pair_idxs=self.PPG_HBB.sx_HBB(rel_pair_idxs[0],proposals)
                        print('num_PPG_after:',len(rel_pair_idxs[0]))

        

    

        if (self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX and self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL)  and ( self.pre == "RPCM" or self.pre == "HetSGG_Predictor" ) :

            device = features[0].device
            for proposal in proposals:
                obj_labels = proposal.get_field("labels")
                proposal.add_field("predict_logits", to_onehot(obj_labels, self.num_obj_cls))
                proposal.add_field("pred_scores", torch.ones(len(obj_labels)).to(device))
                proposal.add_field("pred_labels", obj_labels.to(device))


        if "HBB" in self.type  or "OBB" in self.type:

            rbox = []
            for kk in range(len(proposals)):
                rbox.append(proposals[kk].bbox)
            # 提取特征
            rois = self.rbbox2roi(rbox)
            roi_fea = OBj.bbox_roi_extractor(
                features[:OBj.bbox_roi_extractor.num_inputs], rois)
            roi_features = OBj.bbox_head(roi_fea,flag = True)   ## nums,1024
        else:
            roi_features = self.box_feature_extractor(features, proposals)

        if self.cfg.MODEL.ATTRIBUTE_ON:
            att_features = self.att_feature_extractor(features, proposals)
            roi_features = torch.cat((roi_features, att_features), dim=-1)



        if self.use_union_box:
            union_features = self.union_feature_extractor(features, proposals, rel_pair_idxs, OBj = OBj)
        else:
            union_features = None

       



        # final classifier that converts the features into predictions
        # should corresponding to all the functions and layers after the self.context class
        # refine_logits, relation_logits, add_losses = self.predictor(proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger)

        refine_logits, relation_logits, add_losses= self.predictor(proposals,
                                                                                rel_pair_idxs,
                                                                                rel_labels,
                                                                                rel_binarys,
                                                                                roi_features,
                                                                                union_features,
                                                                                logger)



        if not self.training :
                if self.type != "CV":
                    proposals[0].add_field("s_f", s_f)
                    result = self.post_processor((relation_logits, refine_logits), rel_pair_idxs, proposals)
                    if self.tasks == "Sgdets":
                        result[0].add_field("target", targets[0])

                else:
                     result = self.post_processor((relation_logits, refine_logits), rel_pair_idxs, proposals)

                return roi_features, result, {}

            
        loss_relation, loss_refine = self.loss_evaluator(proposals, rel_labels, relation_logits, refine_logits, cls_new = cls_new)
            

        if self.cfg.MODEL.ATTRIBUTE_ON and isinstance(loss_refine, (list, tuple)):
            output_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine[0], loss_refine_att=loss_refine[1])
        else:

            output_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine)

                

        output_losses.update(add_losses)

        return roi_features, proposals, output_losses
    

def build_roi_relation_head(cfg, in_channels):
    """
    Constructs a new relation head.
    By default, uses ROIRelationHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIRelationHead(cfg, in_channels)
