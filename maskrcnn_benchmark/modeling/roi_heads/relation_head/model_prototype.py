

import math
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


from maskrcnn_benchmark.modeling.utils import cat
#from maskrcnn_benchmark.data import get_dataset_statistics
from maskrcnn_benchmark.structures.bounding_box import BoxList

from .utils_prototype import uniform_hypersphere
from .utils_motifs import obj_edge_vectors


def ham_dist(x, y):
    """
        x   :   [n1 x C]
        y   :   [n2 x C]
        ret :   [n1 x n2]
    """
    return torch.cdist(x, y, p=0.)


def weighted_ham_dist(x, y, weight):
    """
        x   :   [n1 x C]
        y   :   [n2 x C]
        weight: [C]
        ret :   [n1 x n2]
    """
    if len(x.shape) == 1:
        x = x.unsqueeze(0)
    if len(y.shape) == 1:
        y = y.unsqueeze(0)
    C = len(weight)
    x = x.unsqueeze(1).repeat(1, len(y), 1)
    y = y.unsqueeze(0)
    weighted = torch.logical_xor(x, y) * weight.reshape(1, 1, C)
    return torch.mean(weighted, dim=-1)


def ham_score(label, weight):
    """
        label   :  [n x C]
        weight  :  [C]
        return  :   n
    """
    return torch.sum(label * weight.unsqueeze(0), dim=-1)


class PQLayer(nn.Module):
    def __init__(self, feat_dim, K, alpha=1):
        super().__init__()
        # M 个 prototype 每 k 个一个 sub 类，每个 sub 类的 feature 是 D
        self.feat_dim, self.K, self.D = feat_dim, K, feat_dim
        self.alpha = alpha
        # [K, D]
        preGenerated = F.normalize(
            torch.tensor(uniform_hypersphere(self.D, self.K)).float())
        # [M, K, D]
        codebook = torch.empty(self.K, self.D) * float("inf")
        codebook.copy_(preGenerated)
        self._C = nn.Parameter(codebook, requires_grad=True).cuda()
        nn.init.xavier_uniform_(self._C.data)

    def intra_normalization(self, x):
        return F.normalize(x.view(x.shape[0], self.M, self.D),
                           dim=-1).view(x.shape[0], -1)



class MemoryBanks(nn.Module):
    """
        Memory Bank for all objects
        Actually a shell for each class
    """
    def __init__(self, cfg, max_size, feature_dim, sub_proto_list):
        super().__init__()
        self.cfg = cfg
        self.dim = feature_dim
        self.sub_proto_list = sub_proto_list
        # self.obj_list, self.label_weights = self._get_weights(freq_path)
        # load word embedding
        from maskrcnn_benchmark.data import get_dataset_statistics
        statistics = get_dataset_statistics(self.cfg)
        obj_classes = statistics['obj_classes']
        obj_embed_vecs = obj_edge_vectors(
            obj_classes,
            wv_dir=self.cfg.GLOVE_DIR,
            wv_dim=300,
        )
        self.obj_embed = nn.Embedding(49, 300).requires_grad_(False)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.banks = nn.ModuleDict({str(sub_proto) : MemoryBank(self.cfg, max_size, feature_dim, self.obj_embed).cpu() for sub_proto in sub_proto_list}).cpu()
        # self.banks_ = nn.ModuleDict({str(o):MemoryBank(max_size, feature_dim, self.label_weights[o]) for o in self.obj_list})
        # threshold
        self.confident_threshold = 0.9

    @torch.no_grad()
    def read(self):
        """
            feature     : n x d
            label       : [[...], [......], [.....]]
            obj_label   : [n]
        """
        self.pointer_sizes = [v.pointer for k, v in self.banks.items()]

        feat_list = []
        rel_pair_list = []
        proposals_list = []
        union_list = []
        for size, (pred, bank) in zip(self.pointer_sizes, self.banks.items()):
            feature, proposals, rel_pair_idxs, union_feat = bank.read()
            if size != 0:
                torch.cuda.empty_cache()
                feat_list.append(torch.cat(feature))
                rel_pair_list.append(torch.cat(rel_pair_idxs))
                proposals_list.append(proposals)
                # union_feat = [f.unsqueeze(0) for f in union_feat]
                union_list.append(torch.cat(union_feat))
            else:
                feat_list.append([])
                rel_pair_list.append([])
                proposals_list.append([])
                union_list.append([])
            # else:
            #     feat_list.append(torch.zeros((2, 4096)))
            #     rel_pair_list.append(torch.zeros((1, 2)))

            #     bbox = torch.zeros((2, 4))
            #     size = (800, 600)
            #     new_proposal = BoxList(bbox, size)
            #     new_proposal.add_field("labels", torch.zeros((2,)))
            #     new_proposal.add_field("attributes", torch.zeros((2, 10)))
            #     proposals_list.append([new_proposal])
            #     union_list.append(torch.zeros((1, 4096)))
        # if len(feat_list) != 0:
        # roi_feature = torch.cat(feat_list)
        # rel_pair_idxs = torch.cat(rel_pair_list)
        # union_features = torch.cat(union_list)

        return feat_list, proposals_list, rel_pair_list, union_list
        # return None

    @torch.no_grad()
    def write(self, feature, proposals, rel_pair_idxs, rel_dist, union_feat):
        """
            feature     : n x d
            label       : [[...], [......], [.....]]
            obj_label   : [n]
        """
        # obj-related: feature, targets, proposals (num_objs)
        # rel_related: rel_dist, sub_rel_labels, rel_pair_idxs, union_feat (num_rels)
        assert len(rel_dist) == len(rel_pair_idxs)
        for feat, proposal, pair_idx, rel_logits, union in zip(feature, proposals, rel_pair_idxs, rel_dist, union_feat):
            # if confident (predict 和 gt logits 较高的)
            confident = F.softmax(rel_logits, dim=1) # confident for relation logits
            prob, sub_pred = confident.max(dim=-1) # predicted sub prototype label

            rel_mask = prob > self.confident_threshold
            if not rel_mask.any():
                continue
            # select relation
            selected_sub_pred = sub_pred[rel_mask]
            unique_values, inverse_indices = torch.unique(selected_sub_pred, return_inverse=True, sorted=False)
            split_pred = [selected_sub_pred[inverse_indices == i] for i in range(unique_values.shape[0])]
            split_pred_count = [pred.shape[0] for pred in split_pred]
            selected_pair_idx = pair_idx[rel_mask, :].split(split_pred_count, dim=0)
            selected_union = union[rel_mask, :].split(split_pred_count, dim=0)

            for pred, pair_idx, union in zip(unique_values, selected_pair_idx, selected_union):
                pred = str(int(pred))
                offset = sum([feat.shape[0] for feat in self.banks[pred].feat_buffer])

                # selected_ob j_idx = pair_idx.view(-1).unique()
                # selected_feat = feat[selected_obj_idx, :]

                # selected_bboxes = proposal.bbox[selected_obj_idx, :]
                # size = proposal.size
                # selected_obj_labels = proposal.get_field('labels')[selected_obj_idx]
                # selected_obj_attrs = proposal.get_field('attributes')[selected_obj_idx, :]

                # selected_proposal = BoxList(selected_bboxes, size)
                # selected_proposal.add_field("labels", selected_obj_labels)
                # selected_proposal.add_field("attributes", selected_obj_attrs)

                self.banks[pred].write(feat, proposal, pair_idx, union)

            # # select object
            # selected_pair = selected_pair_idx.view(-1)
            # selected_obj_idx = selected_pair.unique()
            # selected_feat = feat[selected_obj_idx, :]
            # # selected_feat = selected_feat.split(2, dim=0)

            # selected_proposals = []
            # selected_bboxes = proposal.bbox[selected_obj_idx, :]
            # size = proposal.size
            # selected_obj_labels = proposal.get_field('labels')[selected_obj_idx]
            # selected_obj_attrs = proposal.get_field('attributes')[selected_obj_idx, :]

            # selected_proposals = BoxList(selected_bboxes, size)
            # selected_proposals.add_field("labels", selected_obj_labels)
            # selected_proposals.add_field("attributes", selected_obj_attrs)

            # for bbox, obj_label, obj_attr in zip(selected_bboxes, selected_obj_labels, selected_obj_attrs):
            #     if len(bbox.shape) == 1:
            #         bbox = bbox.unsqueeze(0)
            #     new_proposal = BoxList(bbox, size)
            #     new_proposal.add_field("labels", obj_label)
            #     new_proposal.add_field("attributes", obj_attr)

            #     selected_proposals.append(new_proposal)

            # 找到分数大于阈值的 and 把最大的那个找出来，和标签比较
            # * done: obj dim and rel dim matching!
            # todo: obj-level and rel-level 分开存
            # selected_feat_ = selected_feat[selected_pair_idx].view(-1, 4096).split(2, dim=0)
            # for pred, s_feat, s_proposal, s_pair_idx, s_union in zip(selected_sub_pred, selected_pair_idx, selected_union):
            #     offset = self.banks[str(int(pred))].pointer
            #     self.banks[str(int(pred))].write(s_feat, s_proposal, s_pair_idx + offset, s_union)

        self.pointer_sizes = [v.pointer for k, v in self.banks.items()]


class MemoryBank(nn.Module):
    """
        Memory Bank for each Object
    """

    def __init__(self, cfg, max_size, feature_dim, obj_embed):
        """
            max_size:       int, the maximum size of the feature bank
            feature_dim:    int, the dimension of feature
            label_weight:   [float], the weight of each verb associated with this object
        """

        super().__init__()
        self.cfg = cfg
        self.obj_embed = obj_embed
        # self.C = len(label_weights)
        # self.label_weights_np = label_weights
        # idx = 2 if len(label_weights) > 5 else 1
        # if idx >= len(label_weights): # ugly if else to be compatible with vcoco
        # self.threshold = 0
        # else:
        # self.threshold = sorted(self.label_weights_np)[idx]

        self.feat_buffer = []
        self.pair_buffer = []
        self.union_buffer = []
        self.proposals_buffer = []

        self.max_size = max_size
        self.pointer = 0
        self.k = 1

        # threshold
        self.diverse_threshold = 0.5

    def _push_like_queue(self, tensor, x, idx=None):
        if idx is None:
            tensor[:-1] = tensor[1:]
            tensor[-1] = x
        tensor[idx] = x
        return tensor

    def read(self):
        return self.feat_buffer, self.proposals_buffer, self.pair_buffer, self.union_buffer

    @torch.no_grad()
    def write(self, feature, proposal, rel_pair_idx, union_feat):
        n_data = 1

        if self.pointer < self.max_size: # if full
            # no full, save the feature nad pair idx
            n_can_store = self.max_size - self.pointer
            if self.pointer + n_data < self.max_size:
                # still have space, just append!
                # self.feat_buffer.append(feature)
                # self.proposals_buffer.append(proposal)
                # self.pair_buffer.append(rel_pair_idx)
                # self.union_buffer.append(union_feat)
                self.feat_buffer[self.pointer:self.pointer+n_data] = feature.unsqueeze(0).detach().clone().cpu()
                self.pair_buffer[self.pointer:self.pointer+n_data] = rel_pair_idx.unsqueeze(0).detach().clone().cpu()
                self.proposals_buffer.append(proposal.to("cpu"))
                self.union_buffer[self.pointer:self.pointer+n_data] = union_feat.unsqueeze(0).detach().clone().cpu()

                # self.ham_score_buffer[self.pointer:self.pointer+n_data] = ham_score(label, self.label_weights).clone()
                self.pointer += n_data
            else:
                self.feat_buffer[self.pointer:self.pointer+n_can_store] = feature.unsqueeze(0).detach().clone().cpu()
                self.pair_buffer[self.pointer:self.pointer+n_can_store] = rel_pair_idx.unsqueeze(0).detach().clone().cpu()
                self.proposals_buffer.append(proposal.to("cpu"))
                self.union_buffer[self.pointer:self.pointer+n_can_store] = union_feat.unsqueeze(0).detach().clone().cpu()
                # self.ham_score_buffer[self.pointer:self.pointer+n_can_store] = ham_score(label, self.label_weights).clone()
                self.pointer = self.max_size
        else:
            # full
            # * if diverse (比较标签对应的 word embedding 的距离)
            # pair_idx = torch.cat(self.pair_buffer)
            target_obj_list = torch.cat([m.get_field("labels") for m in self.proposals_buffer]).unique().cpu()

            target_sub_obj_embed = self.obj_embed(target_obj_list.long())
            # target_sub_obj_embed = target_sub_obj_embed.view(-1, 600)

            obj_list = proposal.get_field("labels").unique().cpu()
            sub_obj_embed = self.obj_embed(obj_list.long())
            # diverse = F.pairwise_distance(sub_obj_embed, target_sub_obj_embed, p=2)
            diverse = torch.stack([F.pairwise_distance(obj, target_sub_obj_embed, p=2) for obj in sub_obj_embed]).mean(dim=0)

            norm_diverse = (diverse - diverse.min()) / (diverse.max() - diverse.min())
            norm_diverse_mask = norm_diverse > self.diverse_threshold
            if not norm_diverse_mask.all():
                return
            replaced_idx = norm_diverse[norm_diverse_mask].argmin()

            if norm_diverse_mask.any():
                # drop the oldest feature and save a new one
                self.feat_buffer = self._push_like_queue(self.feat_buffer, feature.detach().clone().cpu())
                self.pair_buffer = self._push_like_queue(self.pair_buffer, rel_pair_idx.detach().clone().cpu())
                self.proposals_buffer = self._push_like_queue(self.proposals_buffer, proposal)
                self.union_buffer = self._push_like_queue(self.union_buffer, union_feat.detach().clone().cpu())
                # self.feat_buffer = self._push_like_queue(self.feat_buffer, feature.detach().clone().cpu(), replaced_idx)
                # self.pair_buffer = self._push_like_queue(self.pair_buffer, rel_pair_idx.detach().clone().cpu(), replaced_idx)
                # self.proposals_buffer = self._push_like_queue(self.proposals_buffer, proposal, replaced_idx)
                # self.union_buffer = self._push_like_queue(self.union_buffer, union_feat.detach().clone().cpu(), replaced_idx)

        # if self.pointer < self.max_size: # if full
        #     n_can_store = self.max_size - self.pointer
        #     if self.pointer + n_data < self.max_size:
        #         # still many space, just append!
        #         self.feat_buffer[self.pointer:self.pointer+n_data, :] = feature.detach().clone()
        #         self.label_buffer[self.pointer:self.pointer+n_data, :] = label.clone()
        #         self.ham_score_buffer[self.pointer:self.pointer+n_data] = ham_score(label, self.label_weights).clone()
        #         # self.age_buffer[self.pointer:self.pointer+n_data, 0] = n_epoch
        #         # self.age_buffer[self.pointer:self.pointer+n_data, 1] = n_iter
        #         self.pointer += n_data
        #     else:
        #         self.feat_buffer[self.pointer:self.pointer+n_can_store, :] = feature.detach().clone()
        #         self.label_buffer[self.pointer:self.pointer+n_can_store, :] = label.clone()
        #         self.ham_score_buffer[self.pointer:self.pointer+n_can_store] = ham_score(label, self.label_weights).clone()
        #         # self.age_buffer[self.pointer:self.pointer+n_can_store, 0] = n_epoch
        #         # self.age_buffer[self.pointer:self.pointer+n_can_store, 1] = n_iter
        #         self.pointer = self.max_size

        # else:
        #     ham_score_ = ham_score(label, self.label_weights)
        #     if ham_score_ < self.threshold:
        #         return

        #     age_score = self.age_buffer[:, 0] * 10000 + self.age_buffer[:, 1]
        #     _, old_indices  = torch.topk(-age_score, self.pointer)

        #     scores = torch.zeros(self.pointer).to(self.age_buffer.device)
        #     scores[old_indices] += torch.arange(self.pointer).to(self.age_buffer.device)
        #     _, indices = torch.topk(-scores, n_data)

        #     self.feat_buffer[indices, :] = feature.detach().clone()
        #     self.label_buffer[indices, :] = label.clone()
        #     self.ham_score_buffer[indices] = ham_score(label, self.label_weights).clone()
        #     # self.age_buffer[indices, 0] = n_epoch
        #     # self.age_buffer[indices, 1] = n_iter


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        return self.net(x)


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q (bsz, len_q, dim_q)
            k (bsz, len_k, dim_k)
            v (bsz, len_v, dim_v)
            Note: len_k==len_v, and dim_q==dim_k
        Returns:
            output (bsz, len_q, dim_v)
            attn (bsz, len_q, len_k)
        """
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q (bsz, len_q, dim_q)
            k (bsz, len_k, dim_k)
            v (bsz, len_v, dim_v)
            Note: len_k==len_v, and dim_q==dim_k
        Returns:
            output (bsz, len_q, d_model)
            attn (bsz, len_q, len_k)
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size() # len_k==len_v

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

from maskrcnn_benchmark.modeling import registry
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn import functional as F
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.modeling.make_layers import make_fc
from .utils_motifs import obj_edge_vectors, to_onehot, nms_overlaps, encode_box_info
from .model_transformer import TransformerEncoder

class VectorFeature(nn.Module):
    def __init__(self, config, obj_classes, rel_classes, in_channels):
        super(VectorFeature, self).__init__()
        self.cfg = config
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        self.embed_dim = 300  # self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM  # 200
        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)

        self.in_channels = in_channels
        self.obj_dim = in_channels
        # self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM  # 512
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.obj_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.OBJ_LAYER
        self.edge_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM

        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)  # [151, 200]
        self.obj_embed2 = nn.Embedding(self.num_obj_classes, self.embed_dim)  # [151. 200]
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)  # 使用GloVe进行赋值
            self.obj_embed2.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])

        # self.context_obj = TransformerEncoder(self.obj_layer, self.num_head, self.k_dim,
        #                                         self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)

        self.lin_obj = make_fc(self.in_channels + 128, self.hidden_dim)
        self.out_obj = make_fc(self.hidden_dim, self.num_obj_classes)  # 512 -> 151
        self.lin_obj_cyx = make_fc(self.obj_dim + self.embed_dim + 128, self.hidden_dim)

    def forward(self, roi_features, proposals, rel_pair_idxs, logger=None):

        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None

        # label/logits embedding will be used as input
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed1(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight

        assert proposals[0].mode == 'xyxy'
        pos_embed = self.pos_embed(encode_box_info(proposals))

        obj_pre_rep = cat((roi_features, pos_embed), dim=-1)
        num_objs = [len(p) for p in proposals]
        obj_pre_rep = self.lin_obj(obj_pre_rep)  #

        obj_pre_rep_for_pred = self.lin_obj_cyx(
            cat([roi_features, obj_embed, pos_embed], -1))  # 4096 + 128 + 200 -> 512

        if self.mode == 'predcls':
            obj_labels = obj_labels.long()
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_classes)
        else:
            obj_dists = self.out_obj(obj_pre_rep_for_pred)  # 512 -> 151
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs)
            else:
                obj_preds = obj_dists[:, 1:].max(1)[1] + 1

        return obj_dists, obj_preds.long(), obj_pre_rep

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh  # (#box, #box, #class)

            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0  # This way we won't re-sample

            obj_preds.append(out_label.long())
        obj_preds = torch.cat(obj_preds, dim=0)
        return obj_preds