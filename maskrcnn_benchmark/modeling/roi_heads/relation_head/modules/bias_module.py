# -#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author :   Ch
# File    :   bias_module.py
# @Time   :   2021/6/3 10:20
import numpy as np
import torch
import torch.nn as nn

from .utils import load_data


class PenaltyModule(nn.Module):
    def __init__(self, cfg, statistics, penalty_type, fusion_weight):
        super(PenaltyModule, self).__init__()
        self.penalty_threshold = cfg.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.PENALTY_BIAS.PENALTY_THRESHOLD
        self.penalty_weight = cfg.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.PENALTY_BIAS.PENALTY_WEIGHT
        self.scale_weight = cfg.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.PENALTY_BIAS.SCALE_WEIGHT
        self.penalty_type = penalty_type
        self.fusion_weight = fusion_weight

        self.only_RTPB = cfg.only_RTPB

        self.eval_with_penalty = cfg.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.PENALTY_BIAS.EVAL_WITH_PENALTY

        self.penalty_k = cfg.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.PENALTY_BIAS.PENALTY_K
        self.eps = cfg.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.PENALTY_BIAS.PENALTY_EPSILON

        self.weight_path = cfg.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.PENALTY_BIAS.WEIGHT_PATH

        # default value for psb bias
        self.psb_default_value = cfg.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.PENALTY_BIAS.POSSIBLE_BIAS_DEFAULT_VALUE
        if self.psb_default_value <= 0:
            self.psb_default_value = self.eps
        # default value for bg rel
        self.bg_default_value = cfg.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.PENALTY_BIAS.BG_DEFAULT_VALUE
        if self.bg_default_value <= 0:
            self.bg_default_value = self.eps
        self.psb_threshold = cfg.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.PENALTY_BIAS.POSSIBLE_BIAS_THRESHOLD

        self.fg_matrix = statistics['fg_matrix'].detach().clone()
        self.fg_matrix[:, :, 0] = 0

        counts = self.fg_matrix.sum((0, 1))
        count_sum = counts.sum()

        if self.eps < 0:
            # use predefined epsilon value by set it as specific negative value
            if self.eps == -1:
                self.eps = 1 / count_sum
            elif self.eps == -2:
                mean_count = counts[counts.gt(0)].float().mean().item()
                self.eps = 1 / mean_count
            else:
                raise Exception("invalid epsilon value {}".format(self.eps))

        # penalty
        self.penalty_bias = None
        if 'log_bias' == self.penalty_type:
            loaded_data = load_data(self.weight_path)
            loaded_data = loaded_data / loaded_data.sum()
            if self.scale_weight != 1:
                loaded_data = loaded_data.pow(self.scale_weight)
                loaded_data = loaded_data / loaded_data.sum()
            loaded_data = torch.clamp(loaded_data, min=self.eps)
            self.penalty_bias = torch.log(loaded_data)
            # set bg as mean
            # self.penalty_bias[0] = np.log(1 / loaded_data.size()[0] + self.eps)
            self.penalty_bias[0] = np.log(self.bg_default_value)
        elif 'cb_cls' == self.penalty_type:
            dist = counts / count_sum
            if self.scale_weight != 1:
                dist = dist.pow(self.scale_weight)
                dist = dist / dist.sum()
            self.penalty_bias_1 = torch.log(dist + self.eps)

            loaded_data = load_data(self.weight_path)
            if self.scale_weight != 1:
                loaded_data = loaded_data.pow(self.scale_weight)
                loaded_data = loaded_data / loaded_data.sum(-1).view(-1, 1)
            loaded_data = torch.clamp(loaded_data, min=self.eps)
            self.penalty_bias_2 = torch.log(loaded_data)
        elif 'count_bias' == self.penalty_type:
            dist = counts / counts.sum()
            dist[0] = 1
            if self.scale_weight != 1:
                dist = dist.pow(self.scale_weight)
                dist = dist / (dist.sum() - 1)
            self.penalty_bias = torch.log(dist + self.eps)
            self.penalty_bias[0] = np.log(self.bg_default_value)
        elif 'mean_count_bias' == self.penalty_type:
            pair_count = self.fg_matrix.gt(0).sum(0).sum(0).view(-1)
            pair_count[pair_count.eq(0)] = 1
            mean_count = counts / pair_count
            dist = mean_count / pair_count.sum()
            if self.scale_weight != 1:
                dist = dist.pow(self.scale_weight)
                dist = dist / dist.sum()

            self.penalty_bias = torch.log(dist + self.eps)
            # set bg as default
            self.penalty_bias[0] = np.log(self.bg_default_value)
            pass
        elif 'pair_count_bias' == self.penalty_type:
            pair_count = self.fg_matrix.gt(0).sum(0).sum(0).view(-1)
            dist = pair_count / pair_count.sum()
            if self.scale_weight != 1:
                dist = dist.pow(self.scale_weight)
                dist = dist / dist.sum()

            self.penalty_bias = torch.log(dist + self.eps)
            self.penalty_bias[0] = np.log(self.bg_default_value)
        elif 'margin_loss' == self.penalty_type:
            max_margin = 0.5
            scale = 10
            delta = counts.pow(0.25)

            delta[delta.eq(0)] = 1
            margin = 1 / delta
            margin = margin * (max_margin / margin.max())

            self.penalty_bias = - scale * margin

        if 'psb_pair' == self.penalty_type:
            # edit impossible item by pair
            self.fg_count = self.fg_matrix.view(-1, self.fg_matrix.size(-1))
            self.num_obj = self.fg_matrix.size(0)
        if 'psb_sppo' == self.penalty_type:
            #  subj-pred and pred-obj
            self.sp_count = self.fg_matrix.sum(1)
            self.po_count = self.fg_matrix.sum(0)

    def penalty(self, pred_dist, gt=None, obj_pair_label=None):
        resistance_bias = None
        if 'log_bias' == self.penalty_type:
            if self.penalty_bias.device != pred_dist.device:
                self.penalty_bias = self.penalty_bias.to(pred_dist.device)
            resistance_bias = self.penalty_bias
        elif self.penalty_type == 'count_bias' or self.penalty_type == 'pair_count_bias':
            if pred_dist is not None:
                if self.penalty_bias.device != pred_dist.device:
                    self.penalty_bias = self.penalty_bias.to(pred_dist.device)
                resistance_bias = self.penalty_bias
            else:
                pred_dist = torch.zeros(self.penalty_bias)
                resistance_bias = self.penalty_bias
        elif self.penalty_type == 'cb_cls':
            if pred_dist is not None:
                if self.penalty_bias.device != pred_dist.device:
                    self.penalty_bias = self.penalty_bias.to(pred_dist.device)
                resistance_bias = self.penalty_bias[gt]
            else:
                resistance_bias = self.penalty_bias[gt]
                pred_dist = torch.zeros(resistance_bias)
        elif self.penalty_type == 'as_zero':
            return torch.zeros_like(pred_dist)
        elif self.penalty_type == 'psb_pair':
            counts = self.fg_count[(obj_pair_label[:, 0] * self.num_obj + obj_pair_label[:, 1]).long()]
            resistance_bias = torch.log(counts / (counts.sum(1).view(-1, 1) + self.eps) + self.eps)

            resistance_bias[counts.eq(0)] = np.log(self.psb_default_value)
            # bg
            resistance_bias[:, 0] = np.log(self.bg_default_value)
        elif self.penalty_type == 'psb_sppo':
            counts = (self.sp_count[obj_pair_label[:, 0].long()] * self.po_count[obj_pair_label[:, 1].long()]).sqrt()
            resistance_bias = torch.log(counts / (counts.sum(1).view(-1, 1) + self.eps) + self.eps)

            resistance_bias[counts.eq(0)] = np.log(self.psb_default_value)
            # bg
            resistance_bias[:, 0] = np.log(self.bg_default_value)
        elif 'margin_loss' == self.penalty_type:
            resistance_bias = torch.zeros_like(self.penalty_bias).repeat(gt.size(0), 1)
            resistance_bias[torch.arange(gt.size(0)), gt] = self.penalty_bias[gt]
        elif self.penalty_type == '' or self.penalty_type == 'none':
            pass
        else:
            raise Exception('unknown penalty type {}'.format(self.penalty_type))

        if resistance_bias.device != pred_dist.device:
            resistance_bias = resistance_bias.to(pred_dist.device)
        
        if self.only_RTPB:
            return resistance_bias
        else:
            return pred_dist + resistance_bias * self.fusion_weight

    def forward(self, pred_dist, gt=None, obj_pair=None):
        if self.training or self.eval_with_penalty:
            return self.penalty(pred_dist, gt, obj_pair)
        return pred_dist


def make_penalty_modules(cfg, statistics):
    penalty_type = cfg.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.PENALTY_BIAS.PENALTY_TYPE
    penalty_fusion_weights = cfg.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.PENALTY_BIAS.PENALTY_FUSION_WEIGHTS
    penalty_list = [pe for pe in penalty_type.split(';') if pe]
    assert len(penalty_fusion_weights) >= len(penalty_list)

    penalty_modules = nn.ModuleList()
    for i in range(len(penalty_list)):
        pe_type = penalty_list[i]
        if pe_type == 'cb':
            pe_type = 'count_bias'
        weight = penalty_fusion_weights[i]
        # for pe_type in penalty_list:
        penalty_modules.append(PenaltyModule(cfg, statistics, penalty_type=pe_type, fusion_weight=weight))
    return penalty_modules


class FrequencyBias(nn.Module):
    """
    The goal of this is to provide a simplified way of computing
    P(predicate | obj1, obj2, img).
    """

    def __init__(self, cfg, statistics):
        super(FrequencyBias, self).__init__()

        self.eps = cfg.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.EPSILON

        pred_dist = statistics['pred_dist'].float()
        assert pred_dist.size(0) == pred_dist.size(1)
        self.num_objs = pred_dist.size(0)
        self.num_rels = pred_dist.size(2)
        pred_dist = pred_dist.view(-1, self.num_rels)

        self.obj_baseline = nn.Embedding(self.num_objs * self.num_objs, self.num_rels)
        with torch.no_grad():
            self.obj_baseline.weight.copy_(pred_dist, non_blocking=True)

    def index_with_labels(self, obj_pair_labels):
        """
        :param obj_pair_labels: [batch_size, 2]
        :return:
        """
        pair_idx = obj_pair_labels[:, 0] * self.num_objs + obj_pair_labels[:, 1]
        pred_dist = self.obj_baseline(pair_idx.long())
        return pred_dist

    def index_with_probability(self, pair_prob):
        """
        :param labels: [batch_size, num_obj, 2]
        :return:
        """
        batch_size, num_obj, _ = pair_prob.shape

        joint_prob = pair_prob[:, :, 0].contiguous().view(batch_size, num_obj, 1) * pair_prob[:, :,
                                                                                    1].contiguous().view(batch_size, 1,
                                                                                                         num_obj)

        pred_dist = joint_prob.view(batch_size, num_obj * num_obj) @ self.obj_baseline.weight

        return pred_dist

    def forward(self, labels):
        # implement through index_with_labels
        return self.index_with_labels(labels)


class BiasModule(nn.Module):
    def __init__(self, cfg, statistics):
        super(BiasModule, self).__init__()
        # #### load params
        # post operation
        self.use_penalty = cfg.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.USE_PENALTY
        self.dropout_rate = cfg.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.DROPOUT
        # #### all modules
        self.bias_module = None
        self.penalty_module = None
        self.dropout = None
        # #### init post operation
        if self.use_penalty:
            self.penalty_modules = make_penalty_modules(cfg, statistics)
        if self.dropout_rate != 0:
            self.dropout = nn.Dropout(self.dropout_rate)

        # #### init bias module
        # To be done by sub-class
        #
        #
        # #### end

    def before(self, x):
        return x

    def post(self, bias, gt=None, obj_pair=None):
        if self.use_penalty:
            for penalty_module in self.penalty_modules:
                bias = penalty_module(bias, gt=gt, obj_pair=obj_pair)
        if self.dropout_rate != 0 and bias is not None:
            bias = self.dropout(bias)
        return bias

    def forward(self, gt=None, *args, **kwargs):
        bias = None
        bias = self.post(bias, gt=gt)
        return bias


class FreqBiasModule(BiasModule):
    def __init__(self, cfg, statistics):
        super(FreqBiasModule, self).__init__(cfg, statistics)
        self.use_bias = cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        assert self.use_bias
        self.bias_module = FrequencyBias(cfg, statistics)

    def index_with_labels(self, obj_pair_labels, gt=None):
        bias = self.bias_module.index_with_labels(obj_pair_labels)
        bias = self.post(bias, gt=gt, obj_pair=obj_pair_labels)
        return bias

    def index_with_probability(self, pair_prob, gt=None):
        bias = self.bias_module.index_with_probability(pair_prob)
        bias = self.post(bias, gt=gt)
        return bias

    def forward(self, obj_pair_labels=None, gt=None, obj_pair=None, *args, **kwargs):
        return self.index_with_labels(obj_pair_labels=obj_pair_labels, gt=gt)


def build_bias_module(cfg, statistics):
    use_bias = cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
    if use_bias:
        return FreqBiasModule(cfg, statistics)
    else:
        return BiasModule(cfg, statistics)