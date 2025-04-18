import torch

from .box_head.box_head import build_roi_box_head
from .mask_head.mask_head import build_roi_mask_head
from .attribute_head.attribute_head import build_roi_attribute_head
from .keypoint_head.keypoint_head import build_roi_keypoint_head
from .relation_head.relation_head import build_roi_relation_head
import json

class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        self.type = self.cfg.Type 
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.KEYPOINT_ON and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.keypoint.feature_extractor = self.box.feature_extractor


        if self.cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR in ['HetSGG_Predictor', 'HetSGGplus_Predictor']:
            self.vg_cat_dict = json.load(open(f'/media/dell/data1/WTZ/SGG_Frame/data/labels.json', 'r'))

        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            self.mode = 'predcls' if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL else "sgcls"
        else:
            self.mode = "sgdet"

    def compute_category(self, predictions):
        if self.mode != 'predcls':
            if self.cfg.MODEL.ROI_RELATION_HEAD.HETSGG.CLASS_AGG == 'sum':
                for i, _ in enumerate(predictions):
                    prod_except_background = self.vg_cat_dict['catidx_labelgroup'][str(0)][:-1]
                    product_score = torch.softmax(predictions[i].extra_fields['predict_logits'], 1)[:,prod_except_background].sum(1).view(-1,1)
                    human_score = torch.softmax(predictions[i].extra_fields['predict_logits'], 1)[:,self.vg_cat_dict['catidx_labelgroup'][str(1)]].sum(1).view(-1,1)
                    # animal_score = torch.softmax(predictions[i].extra_fields['predict_logits'], 1)[:,self.vg_cat_dict['catidx_labelgroup'][str(2)]].sum(1).view(-1,1)
                    predictions[i].extra_fields['category_scores'] = torch.cat([product_score, human_score], dim = 1)
                    
            elif self.cfg.MODEL.ROI_RELATION_HEAD.HETSGG.CLASS_AGG== 'mean':
                for i, _ in enumerate(predictions):
                    prod_except_background = self.vg_cat_dict['catidx_labelgroup'][str(0)][:-1]
                    product_score = torch.softmax(predictions[i].extra_fields['predict_logits'], 1)[:,prod_except_background].mean(1).view(-1,1)
                    human_score = torch.softmax(predictions[i].extra_fields['predict_logits'], 1)[:,self.vg_cat_dict['catidx_labelgroup'][str(1)]].mean(1).view(-1,1)
                    # animal_score = torch.softmax(predictions[i].extra_fields['predict_logits'], 1)[:,self.vg_cat_dict['catidx_labelgroup'][str(2)]].mean(1).view(-1,1)
                    predictions[i].extra_fields['category_scores'] = torch.cat([product_score, human_score], dim = 1)

   # def forward(self, features, proposals, targets=None, logger=None):
    def forward(self, features, proposals, targets=None, logger=None, ite=None,OBj = None,s_f = None,m = None,MUL =None,val = None,vae = None, bce = None):
        losses = {}
        if self.type != "CV":
            cls_new = None
            detections = proposals
        else:
            x, detections, loss_box = self.box(features, proposals, targets,logger = logger)

        if not self.cfg.MODEL.RELATION_ON:
            # During the relationship training stage, the bbox_proposal_network should be fixed, and no loss. 
            losses.update(loss_box)

        if self.cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR in ['HetSGG_Predictor', 'HetSGGplus_Predictor'] and self.mode != 'predcls':
            self.compute_category(detections)
            
        if self.cfg.MODEL.ATTRIBUTE_ON:
            # Attribute head don't have a separate feature extractor
            z, detections, loss_attribute = self.attribute(features, detections, targets)
            losses.update(loss_attribute)

        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_mask = self.mask(mask_features, detections, targets)
            losses.update(loss_mask)

        if self.cfg.MODEL.KEYPOINT_ON:
            keypoint_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                keypoint_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_keypoint = self.keypoint(keypoint_features, detections, targets)
            losses.update(loss_keypoint)

        if self.cfg.MODEL.RELATION_ON:
            # it may be not safe to share features due to post processing
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
          #  x, detections, loss_relation = self.relation(features, detections, targets, logger)
            if self.cfg.CFA_pre == 'extract_aug':
                tail_dict = self.relation(features, detections, targets, logger,OBj= OBj)
                return tail_dict
            
    
            x, detections, loss_relation = self.relation(features, detections, targets, logger,
                                                         ite=ite,OBj= OBj,s_f = s_f,MUL =MUL,m=m,val = val, vae = vae, bce = bce)
            losses.update(loss_relation)

        return x, detections, losses


def build_roi_heads(cfg, in_channels,RS = None,RS_conf = None):
    # individually create the heads, that will be combined together
    # afterwards


    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []

    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg, in_channels)))
    if cfg.MODEL.KEYPOINT_ON:
        roi_heads.append(("keypoint", build_roi_keypoint_head(cfg, in_channels)))
    if cfg.MODEL.RELATION_ON:
        roi_heads.append(("relation", build_roi_relation_head(cfg, in_channels)))
    if cfg.MODEL.ATTRIBUTE_ON:
        roi_heads.append(("attribute", build_roi_attribute_head(cfg, in_channels)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads
