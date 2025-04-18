# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os.path as path

import torch
import torch.nn as nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.modeling.roi_heads.relation_head.modules.utils import load_data
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.config import cfg

class RelationLossComputation(object):
    """
    Computes the loss for relation triplet.
    Also supports FPN
    """

    def __init__(
            self,
            attri_on,
            num_attri_cat,
            max_num_attri,
            attribute_sampling,
            attribute_bgfg_ratio,
            use_label_smoothing,
            predicate_proportion,
            use_focal_loss=False,
            focal_loss_param=None,
            weight_path=''
    ):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """
        if focal_loss_param is None:
            focal_loss_param = {}
        self.attri_on = attri_on
        self.num_attri_cat = num_attri_cat
        self.max_num_attri = max_num_attri
        self.attribute_sampling = attribute_sampling
        self.attribute_bgfg_ratio = attribute_bgfg_ratio
        self.use_label_smoothing = use_label_smoothing
        # self.pred_weight = (1.0 / torch.FloatTensor([0.5,] + predicate_proportion)).cuda()
        if self.use_label_smoothing:
            self.criterion_loss = Label_Smoothing_Regression(e=0.01)
        else:
            self.criterion_loss = nn.CrossEntropyLoss()

        loss_weight = None
        if weight_path and path.exists(weight_path):
            loss_weight = load_data(weight_path)
            loss_weight = loss_weight.cuda()

        if use_focal_loss:
            self.rel_criterion_loss = FocalLoss(**focal_loss_param)
        else:
            self.rel_criterion_loss = nn.CrossEntropyLoss(weight=loss_weight)

    def __call__(self, proposals, rel_labels, relation_logits,
                     refine_logits, relation_logits1=None, r1=None, r2=None,cls_new = None, cur_iter=None, rel_pair_idxs=None, all_changed_classes=None, all_change_to_classes=None):
        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.

        Arguments:
            relation_logits (list[Tensor])
            refine_obj_logits (list[Tensor])

        Returns:
            predicate_loss (Tensor)
            finetune_obj_loss (Tensor)
        """
        if self.attri_on:
            if isinstance(refine_logits[0], (list, tuple)):
                refine_obj_logits, refine_att_logits = refine_logits
            else:
                # just use attribute feature, do not actually predict attribute
                self.attri_on = False
                refine_obj_logits = refine_logits
        else:
            refine_obj_logits = refine_logits

        relation_logits = cat(relation_logits, dim=0)
        refine_obj_logits = cat(refine_obj_logits, dim=0)

        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)

        rel_labels = cat(rel_labels, dim=0)


        loss_relation = self.rel_criterion_loss(relation_logits, rel_labels.long())

        loss_refine_obj = self.criterion_loss(refine_obj_logits, fg_labels.long())

            

        # The following code is used to calculate sampled attribute loss
        if self.attri_on:
            refine_att_logits = cat(refine_att_logits, dim=0)
            fg_attributes = cat([proposal.get_field("attributes") for proposal in proposals], dim=0)

            attribute_targets, fg_attri_idx = self.generate_attributes_target(fg_attributes)
            if float(fg_attri_idx.sum()) > 0:
                # have at least one bbox got fg attributes
                refine_att_logits = refine_att_logits[fg_attri_idx > 0]
                attribute_targets = attribute_targets[fg_attri_idx > 0]
            else:
                refine_att_logits = refine_att_logits[0].view(1, -1)
                attribute_targets = attribute_targets[0].view(1, -1)

            loss_refine_att = self.attribute_loss(refine_att_logits, attribute_targets,
                                                  fg_bg_sample=self.attribute_sampling,
                                                  bg_fg_ratio=self.attribute_bgfg_ratio)
            return loss_relation, (loss_refine_obj, loss_refine_att)
        else:

             return loss_relation, loss_refine_obj

    def generate_attributes_target(self, attributes):
        """
        from list of attribute indexs to [1,0,1,0,0,1] form
        """
        assert self.max_num_attri == attributes.shape[1]
        device = attributes.device
        num_obj = attributes.shape[0]

        fg_attri_idx = (attributes.sum(-1) > 0).long()
        attribute_targets = torch.zeros((num_obj, self.num_attri_cat), device=device).float()

        for idx in torch.nonzero(fg_attri_idx).squeeze(1).tolist():
            for k in range(self.max_num_attri):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1
        return attribute_targets, fg_attri_idx

    def attribute_loss(self, logits, labels, fg_bg_sample=True, bg_fg_ratio=3):
        if fg_bg_sample:
            loss_matrix = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').view(-1)
            fg_loss = loss_matrix[labels.view(-1) > 0]
            bg_loss = loss_matrix[labels.view(-1) <= 0]

            num_fg = fg_loss.shape[0]
            # if there is no fg, add at least one bg
            num_bg = max(int(num_fg * bg_fg_ratio), 1)
            perm = torch.randperm(bg_loss.shape[0], device=bg_loss.device)[:num_bg]
            bg_loss = bg_loss[perm]

            return torch.cat([fg_loss, bg_loss], dim=0).mean()
        else:
            attri_loss = F.binary_cross_entropy_with_logits(logits, labels)
            attri_loss = attri_loss * self.num_attri_cat / 20.0
            return attri_loss

class RelationContraLossComputation(object):
    """
    Computes the loss for relation triplet.
    Also supports FPN
    """

    def __init__(
        self,
        attri_on,
        num_attri_cat,
        max_num_attri,
        attribute_sampling,
        attribute_bgfg_ratio,
        use_label_smoothing,
        predicate_proportion,
    ):
        """
        Arguments:
            bbox_proposal_matcher (Matcher)
            rel_fg_bg_sampler (RelationPositiveNegativeSampler)
        """
        self.attri_on = attri_on
        self.num_attri_cat = num_attri_cat
        self.max_num_attri = max_num_attri
        self.attribute_sampling = attribute_sampling
        self.attribute_bgfg_ratio = attribute_bgfg_ratio
        self.use_label_smoothing = use_label_smoothing
        self.pred_weight = (1.0 / torch.FloatTensor([0.5,] + predicate_proportion)).cuda()
        
        # obtained from train split with filter_non_overlap
        self.pred_cnt = [8415, 273, 7830, 13410, 7252, 6535, 527, 5988, 3326, 1388, 9081, 22638, 541, 704, 433, 1974, 851, 2996, 4733, 54, 10960, 10281, 30094, 1565, 248, 242, 10908, 73, 2089, 1244, 1780, 252, 12961, 1961, 738, 1215, 4215, 4863, 61, 838, 2, 76, 89, 73, 203, 343, 799, 319, 12, 1038, 3685, 753, 39, 34, 264, 689, 53, 6]
       # self.pred_cnt = [6712, 171, 208, 379, 504, 1829, 1413, 10011, 644, 394, 1603, 397, 460, 565, 4, 809, 163, 157, 663, 67144, 10764, 21748, 3167, 752, 676, 364, 114, 234, 15300, 31347, 109355, 333, 793, 151, 601, 429, 71, 4260, 44, 5086, 2273, 299, 3757, 551, 270, 1225, 352, 47326, 4810, 11059]
        self.HEAD_IDS = cfg.HEAD_IDS
        self.pred_loss_type = None
        if cfg.MIXUP.PREDICATE_LOSS_TYPE is not None:
            self.pred_loss_type = cfg.MIXUP.PREDICATE_LOSS_TYPE

        self.use_curri = cfg.MIXUP.PREDICATE_USE_CURRI
        if self.use_curri:
            assert self.pred_loss_type.startswith('CB')

        if self.use_label_smoothing:
            self.criterion_loss = Label_Smoothing_Regression(e=0.01)
        else:
            self.criterion_loss = nn.CrossEntropyLoss()


    def __call__(self, proposals, rel_labels, relation_logits, refine_logits, old_proposals, old_rel_labels, old_relation_logits, old_refine_logits, cur_iter=None, rel_pair_idxs=None, changed_obj_idxs=None, changed_pred_idxs=None, filter_old_tail_idxs=None, filter_tail_idxs=None, all_changed_classes=None, all_change_to_classes=None):
        """
        Computes the loss for relation triplet.
        This requires that the subsample method has been called beforehand.

        Arguments:
            relation_logits (list[Tensor])
            refine_obj_logits (list[Tensor])

        Returns:
            predicate_loss (Tensor)
            finetune_obj_loss (Tensor)
        """
        if self.attri_on:
            if isinstance(refine_logits[0], (list, tuple)):
                refine_obj_logits, refine_att_logits = refine_logits
            else:
                # just use attribute feature, do not actually predict attribute
                self.attri_on = False
                refine_obj_logits = refine_logits
        else:
            refine_obj_logits = refine_logits
            old_refine_obj_logits = old_refine_logits


        sum = 0
        for i in range(len(changed_obj_idxs)):
            changed_obj_idxs[i] = changed_obj_idxs[i] + sum
            sum += refine_logits[i].shape[0]
        
        changed_obj_idxs = cat(changed_obj_idxs, dim=0).type(torch.long)

        relation_logits = cat(relation_logits, dim=0)
        refine_obj_logits = cat(refine_obj_logits, dim=0)
        fg_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        rel_labels = cat(rel_labels, dim=0)

        old_relation_logits = cat(old_relation_logits, dim=0)
        old_refine_obj_logits = cat(old_refine_obj_logits, dim=0)
        old_fg_labels = cat([proposal.get_field("labels") for proposal in old_proposals], dim=0)
        old_rel_labels = cat(old_rel_labels, dim=0)

        if self.pred_loss_type is None:
            loss_relation = self.criterion_loss(relation_logits, rel_labels.long())
        
        elif self.pred_loss_type == 'MIXUP_CE':
            device = refine_logits[0].device
            if rel_labels.shape[0] == 0 and old_rel_labels.shape[0] == 0:
                loss_relation = torch.tensor(0).to(device)

            elif rel_labels.shape[0] != 0 and old_rel_labels.shape[0] == 0:
                loss_relation = F.cross_entropy(input=relation_logits, target=rel_labels)
            else:
                loss_relation = F.cross_entropy(input=relation_logits, target=rel_labels) 

        loss_refine_obj = self.criterion_loss(refine_obj_logits, fg_labels.long()) + self.criterion_loss(old_refine_obj_logits, old_fg_labels.long()) 

        if changed_obj_idxs.shape[0] > 0:
            obj_contra_loss = 0.1 * self.contrastive_loss_forward(refine_obj_logits[changed_obj_idxs], old_refine_obj_logits[changed_obj_idxs])
        else:
            obj_contra_loss = torch.tensor(0).to(device)


        # The following code is used to calcaulate sampled attribute loss
        if self.attri_on:
            refine_att_logits = cat(refine_att_logits, dim=0)
            fg_attributes = cat([proposal.get_field("attributes") for proposal in proposals], dim=0)

            attribute_targets, fg_attri_idx = self.generate_attributes_target(fg_attributes)
            if float(fg_attri_idx.sum()) > 0:
                # have at least one bbox got fg attributes
                refine_att_logits = refine_att_logits[fg_attri_idx > 0]
                attribute_targets = attribute_targets[fg_attri_idx > 0]
            else:
                refine_att_logits = refine_att_logits[0].view(1, -1)
                attribute_targets = attribute_targets[0].view(1, -1)

            loss_refine_att = self.attribute_loss(refine_att_logits, attribute_targets, 
                                             fg_bg_sample=self.attribute_sampling, 
                                             bg_fg_ratio=self.attribute_bgfg_ratio)
            return loss_relation, (loss_refine_obj, loss_refine_att)

        else:
            return loss_relation, loss_refine_obj, obj_contra_loss

    def generate_attributes_target(self, attributes):
        """
        from list of attribute indexs to [1,0,1,0,0,1] form
        """
        assert self.max_num_attri == attributes.shape[1]
        device = attributes.device
        num_obj = attributes.shape[0]

        fg_attri_idx = (attributes.sum(-1) > 0).long()
        attribute_targets = torch.zeros((num_obj, self.num_attri_cat), device=device).float()

        for idx in torch.nonzero(fg_attri_idx).squeeze(1).tolist():
            for k in range(self.max_num_attri):
                att_id = int(attributes[idx, k])
                if att_id == 0:
                    break
                else:
                    attribute_targets[idx, att_id] = 1
        return attribute_targets, fg_attri_idx

    def attribute_loss(self, logits, labels, fg_bg_sample=True, bg_fg_ratio=3):
        if fg_bg_sample:
            loss_matrix = F.binary_cross_entropy_with_logits(logits, labels, reduction='none').view(-1)
            fg_loss = loss_matrix[labels.view(-1) > 0]
            bg_loss = loss_matrix[labels.view(-1) <= 0]

            num_fg = fg_loss.shape[0]
            # if there is no fg, add at least one bg
            num_bg = max(int(num_fg * bg_fg_ratio), 1)   
            perm = torch.randperm(bg_loss.shape[0], device=bg_loss.device)[:num_bg]
            bg_loss = bg_loss[perm]

            return torch.cat([fg_loss, bg_loss], dim=0).mean()
        else:
            attri_loss = F.binary_cross_entropy_with_logits(logits, labels)
            attri_loss = attri_loss * self.num_attri_cat / 20.0
            return attri_loss

    def contrastive_loss_forward(self,
                                  hidden1: torch.Tensor,
                                  hidden2: torch.Tensor,
                                  hidden_norm: bool = True,
                                  temperature: float = 1.0,
                                  LARGE_NUM = 1e9):
        """
        hidden1: (batch_size, dim)
        hidden2: (batch_size, dim)
        """
        batch_size, hidden_dim = hidden1.shape
        
        if hidden_norm:
            hidden1 = torch.nn.functional.normalize(hidden1, p=2, dim=-1)
            hidden2 = torch.nn.functional.normalize(hidden2, p=2, dim=-1)

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = torch.arange(0, batch_size).to(device=hidden1.device)
        masks = torch.nn.functional.one_hot(torch.arange(0, batch_size), num_classes=batch_size).to(device=hidden1.device, dtype=torch.float)

        logits_aa = torch.matmul(hidden1, hidden1_large.transpose(0, 1)) / temperature  # shape (batch_size, batch_size)
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.transpose(0, 1)) / temperature  # shape (batch_size, batch_size)
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.transpose(0, 1)) / temperature  # shape (batch_size, batch_size)
        logits_ba = torch.matmul(hidden2, hidden1_large.transpose(0, 1)) / temperature  # shape (batch_size, batch_size)

        loss_a = torch.nn.functional.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels)
        loss_b = torch.nn.functional.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)
        loss = loss_a + loss_b
        return loss



class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        target = target.view(-1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.index_select(-1, target).diag()
        logpt = logpt.view(-1)
        pt = logpt.exp()

        logpt = logpt * self.alpha * (target > 0).float() + logpt * (1 - self.alpha) * (target <= 0).float()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class SoftTargetLoss(nn.Module):
    def __init__(self, target_path, reduction='mean'):
        super(SoftTargetLoss, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.reduction = reduction
        if isinstance(target_path, str):
            self.target_matrix = torch.load(target_path)
        else:
            self.target_matrix = target_path

        self.target_matrix[0, 0] = 1
        return

    def get_target_vec(self, target):
        target_vec = self.target_matrix[target]
        return target_vec.to(target.device)

    def forward(self, x, target):
        if x.size(0) != target.size(0):
            raise ValueError('Expected input batchsize ({}) to match target batch_size({})'
                             .format(x.size(0), target.size(0)))

        if x.dim() != 2:
            raise ValueError('Only 2 dimension tensor are implemented, (got {})'
                             .format(x.size()))

        target_vec = self.get_target_vec(target)
        x = self.log_softmax(x)
        loss = torch.sum(- x * target_vec, dim=1)

        if self.reduction == 'none':
            return loss

        elif self.reduction == 'sum':
            return torch.sum(loss)

        elif self.reduction == 'mean':
            return torch.mean(loss)

        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')


def make_roi_relation_loss_evaluator(cfg):
    loss_evaluator = RelationLossComputation(
        cfg.MODEL.ATTRIBUTE_ON,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO,
        cfg.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS,
        cfg.MODEL.ROI_RELATION_HEAD.REL_PROP,
        use_focal_loss=cfg.MODEL.ROI_RELATION_HEAD.USE_FOCAL_LOSS,
        focal_loss_param={
            'gamma': cfg.MODEL.ROI_RELATION_HEAD.FOCAL_LOSS.GAMMA,
            'alpha': cfg.MODEL.ROI_RELATION_HEAD.FOCAL_LOSS.ALPHA,
            'size_average': cfg.MODEL.ROI_RELATION_HEAD.FOCAL_LOSS.SIZE_AVERAGE
        },
        weight_path=cfg.MODEL.ROI_RELATION_HEAD.LOSS_WEIGHT_PATH,
    )

    return loss_evaluator

def make_roi_relation_contra_loss_evaluator(cfg):

    loss_evaluator = RelationContraLossComputation(
        cfg.MODEL.ATTRIBUTE_ON,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE,
        cfg.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO,
        cfg.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS,
        cfg.MODEL.ROI_RELATION_HEAD.REL_PROP,
    )

    return loss_evaluator
