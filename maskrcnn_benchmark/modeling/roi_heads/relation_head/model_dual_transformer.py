# -#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author :   Ch
# File    :   model_debug.py
# @Time   :   2021/7/12 15:34
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.roi_heads.relation_head.model_cross_transformer import CrossTransformerEncoder
from maskrcnn_benchmark.modeling.roi_heads.relation_head.model_transformer import TransformerContext, \
    TransformerEncoder
from maskrcnn_benchmark.modeling.utils import cat
from .modules.bias_module import build_bias_module
from .utils_motifs import obj_edge_vectors, to_onehot, nms_overlaps, encode_box_info, encode_orientedbox_info
from .utils_relation import layer_init
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_relation import nms_overlaps_rotated


class GTransformerContext(nn.Module):
    """
        contextual encoding of objects
    """

    def __init__(self, config, obj_classes, rel_classes, in_channels):
        super(GTransformerContext, self).__init__()
        self.cfg = config
        # setting parameters
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            self.mode = 'predcls' if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL else 'sgcls'
        else:
            self.mode = 'sgdet'

        ### change
        in_channels = 4096
        ####
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_cls = len(obj_classes)
        self.num_rel_cls = len(rel_classes)
        self.in_channels = in_channels
        self.obj_dim = in_channels
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.obj_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.OBJ_LAYER
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM

        embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed1 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.obj_embed2 = nn.Embedding(self.num_obj_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(embed_vecs, non_blocking=True)
            self.obj_embed2.weight.copy_(embed_vecs, non_blocking=True)

        # position embedding
        self.bbox_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(32, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
        ])

        embed_dim = self.embed_dim
        self.type = self.cfg.Type
        # for other embed operation

        # ###
        self.lin_obj = nn.Linear(self.in_channels + embed_dim + 128, self.hidden_dim)
        layer_init(self.lin_obj, xavier=True)

        self.out_obj = nn.Linear(self.hidden_dim, self.num_obj_cls)
        self.context_obj = TransformerEncoder(self.obj_layer, self.num_head, self.k_dim,
                                              self.v_dim, self.hidden_dim, self.inner_dim, self.dropout_rate)

    def forward(self, roi_features, proposals, rel_pair_idxs=None, logger=None, ctx_average=False):
        # labels will be used in DecoderRNN during training
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None

        # label/logits embedding will be used as input
        # obj_pred will be use as predicated label
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_embed = self.obj_embed1(obj_labels.long())
            obj_pred = obj_labels
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight
            obj_pred = obj_logits[:, 1:].max(1)[1] + 1

        # bbox embedding will be used as input
        # 'xyxy' --> dim-9 --> fc*2 + ReLU --> dim-128
        # assert proposals[0].mode == 'xyxy'
        if proposals[0].bbox.shape[-1] == 5:
            pos_embed = self.bbox_embed(encode_orientedbox_info(proposals))
        else:
            pos_embed = self.bbox_embed(encode_box_info(proposals))

        # encode objects with transformer
        obj_pre_rep = cat((roi_features, obj_embed, pos_embed), -1)

        num_objs = [len(p) for p in proposals]
        obj_pre_rep = self.lin_obj(obj_pre_rep)
        # graph mask
        graph_mask = None

        obj_feats = self.context_obj(obj_pre_rep, num_objs, graph_mask)

        # predict obj_dists and obj_preds
        if self.mode == 'predcls':
            assert obj_labels is not None
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_cls)
            # edge_pre_rep = cat((roi_features, obj_feats, self.obj_embed2(obj_labels.long())), dim=-1)
        else:
            obj_dists = self.out_obj(obj_feats)
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:  # change RS
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs)
            else:
                 obj_preds = obj_dists[:, 1:].max(1)[1] + 1

          
        return obj_dists, obj_preds, obj_feats, None


    def build_sub_graph_mask(self, obj_labels, num_obj):
        batch_size = len(num_obj)
        padding_size = max(num_obj)
        if self.use_weighted_graph_mask:
            res = np.ndarray((batch_size, padding_size, padding_size),
                             dtype=np.float32)  # batch_size * max_obj_cnt * max_obj_cnt
            res[:, :, :] = -1
            start_index = 0
            for img_idx in range(len(num_obj)):
                img_obj_cnt = num_obj[img_idx]
                for i in range(padding_size):
                    res[img_idx, i, i] = 1
                for i in range(start_index, start_index + img_obj_cnt):
                    for j in range(start_index, start_index + img_obj_cnt):
                        if i == j:
                            continue
                        res[img_idx, i - start_index, j - start_index] = self.graph_mask[obj_labels[i]][
                            obj_labels[j]].item()
                start_index += img_obj_cnt
            res = torch.tensor(res, device=obj_labels.device)
            res = F.softmax(res, dim=1)
            return res
        else:
            res = np.ndarray((batch_size, padding_size, padding_size),
                             dtype=np.bool)  # batch_size * max_obj_cnt * max_obj_cnt
            res[:, :, :] = False

            start_index = 0
            for img_idx in range(len(num_obj)):
                img_obj_cnt = num_obj[img_idx]
                for i in range(padding_size):
                    res[img_idx, i, i] = True
                for i in range(start_index, start_index + img_obj_cnt):
                    for j in range(start_index, start_index + img_obj_cnt):
                        if i == j:
                            continue
                        res[img_idx, i - start_index, j - start_index] = self.graph_mask[obj_labels[i]][
                            obj_labels[j]].item()

                start_index += img_obj_cnt
            return torch.tensor(res, device=obj_labels.device)

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            if "HBB" in self.type:
                is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh  # (#box, #box, #class)
            else:
                is_overlap = nms_overlaps_rotated(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh  # (#box, #box, #class)
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


class BaseTransformerEncoder(nn.Module):
    def __init__(self, input_dim, out_dim, n_layer, num_head, k_dim, v_dim, dropout_rate=0.1,
                 ):
        super(BaseTransformerEncoder, self).__init__()

        self.dropout_rate = dropout_rate

        self.num_head = num_head
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.graph_encoder = TransformerEncoder(n_layer, self.num_head, self.k_dim,
                                                self.v_dim, input_dim, out_dim, self.dropout_rate)

    def forward(self, features, counts, adj_matrices=None):
        """
        Args:
            features: Feature Tensor to be encoded
            counts: count of item of each sample. [batch-size]
            adj_matrices: None for dense connect.
                List of adjustment matrices with:
                Bool(True for connect) or
                Float(negative for not connected pair)
        Returns:
            Encode result

        """
        if adj_matrices is not None:
            adj_matrices = self.build_padding_adj(adj_matrices, counts)
        features = self.graph_encoder(features, counts, adj_matrices)
        return features

    @staticmethod
    def build_padding_adj(adj_matrices, counts):
        """
        expand the adj matrix to the same size, and stack them into one Tensor
        Args:
            adj_matrices:
            counts:

        Returns:

        """
        padding_size = max(counts)
        index = torch.arange(padding_size).long()

        res = []
        for adj in adj_matrices:
            expand_mat = torch.zeros(size=(padding_size, padding_size)) - 1
            expand_mat[index, index] = 1
            expand_mat = expand_mat.to(adj)
            adj_count = adj.size(0)
            expand_mat[:adj_count, :adj_count] = adj
            res.append(expand_mat)

        return torch.stack(res)


@registry.ROI_RELATION_PREDICTOR.register("DualTransPredictor")
class DualTransPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(DualTransPredictor, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.debug_flag = config.DEBUG
        self.eval_fc = config.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.EVAL_USE_FC
        assert in_channels is not None
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION

        # load class dict
        from maskrcnn_benchmark.data import get_dataset_statistics
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        # assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        # ##################### init model #####################
        self.use_gtrans_context = config.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.USE_GTRANS_CONTEXT
        # context layer (message pass)
        if self.use_gtrans_context:
            self.context_layer = GTransformerContext(config, obj_classes, rel_classes, in_channels)
        else:
            self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.edge_repr_dim = self.hidden_dim * 2
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_obj_edge_repr = nn.Linear(self.hidden_dim, self.edge_repr_dim)
        layer_init(self.post_obj_edge_repr, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)

        self.epsilon = 0.001

        self.use_rel_graph = config.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.USE_REL_GRAPH

        # ### graph model
        # use gcn as temp
        self.use_graph_encode = config.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.USE_GRAPH_ENCODE
        self.graph_enc_strategy = config.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.GRAPH_ENCODE_STRATEGY

        if self.use_graph_encode:
            if self.graph_enc_strategy == 'trans':
                # encode relationship with trans
                self.pred_up_dim = nn.Linear(self.edge_repr_dim, self.pooling_dim)
                # ### change
                # if Method_flag == "OBB":
                #     self.mix_ctx = nn.Linear(self.pooling_dim + 1024, self.edge_repr_dim)
                # else:
                self.mix_ctx = nn.Linear(self.pooling_dim * 2, self.edge_repr_dim)
                #self.mix_ctx = nn.Linear(self.pooling_dim * 2, self.edge_repr_dim)

                n_layer = config.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.TRANSFORMER.REL_LAYER
                num_head = config.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.TRANSFORMER.NUM_HEAD
                k_dim = config.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.TRANSFORMER.KEY_DIM
                v_dim = config.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.TRANSFORMER.VAL_DIM
                dropout_rate = config.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.TRANSFORMER.DROPOUT_RATE
                self.graph_encoder = nn.ModuleList(
                    [
                        BaseTransformerEncoder(input_dim=self.edge_repr_dim, out_dim=self.edge_repr_dim,
                                               n_layer=n_layer, num_head=num_head, k_dim=k_dim, v_dim=v_dim,
                                               dropout_rate=dropout_rate)
                    ])
            elif self.graph_enc_strategy == 'cross_trans':
                # encode relationship with cross-Trans between obj and rel
                self.pred_up_dim = nn.Linear(self.edge_repr_dim, self.pooling_dim)
                self.mix_ctx = nn.Linear(self.pooling_dim * 2, self.edge_repr_dim)

                n_layer = config.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.TRANSFORMER.REL_LAYER
                num_head = config.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.TRANSFORMER.NUM_HEAD
                k_dim = config.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.TRANSFORMER.KEY_DIM
                v_dim = config.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.TRANSFORMER.VAL_DIM
                dropout_rate = config.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.TRANSFORMER.DROPOUT_RATE

                self.graph_encoder = nn.ModuleList(
                    [
                        CrossTransformerEncoder(d_model_q=self.edge_repr_dim, d_model_kv=self.hidden_dim,
                                                d_inner=self.edge_repr_dim,
                                                n_layer=n_layer, num_head=num_head, k_dim=k_dim, v_dim=v_dim,
                                                dropout_rate=dropout_rate)
                    ])
            elif self.graph_enc_strategy == 'all_trans':
                # encode relationship with cross-Trans between obj and rel and trans among rel
                self.pred_up_dim = nn.Linear(self.edge_repr_dim, self.pooling_dim)
                self.mix_ctx = nn.Linear(self.pooling_dim * 2, self.edge_repr_dim)

                n_layer = config.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.TRANSFORMER.REL_LAYER
                num_head = config.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.TRANSFORMER.NUM_HEAD
                k_dim = config.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.TRANSFORMER.KEY_DIM
                v_dim = config.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.TRANSFORMER.VAL_DIM
                dropout_rate = config.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.TRANSFORMER.DROPOUT_RATE

                self.graph_encoder_NE = nn.ModuleList(
                    [
                        CrossTransformerEncoder(d_model_q=self.edge_repr_dim, d_model_kv=self.hidden_dim,
                                                d_inner=self.edge_repr_dim,
                                                n_layer=n_layer, num_head=num_head, k_dim=k_dim, v_dim=v_dim,
                                                dropout_rate=dropout_rate)
                    ])
                self.graph_encoder_EE = nn.ModuleList(
                    [
                        BaseTransformerEncoder(input_dim=self.edge_repr_dim, out_dim=self.edge_repr_dim,
                                               n_layer=n_layer, num_head=num_head, k_dim=k_dim, v_dim=v_dim,
                                               dropout_rate=dropout_rate)
                    ])
            elif self.graph_enc_strategy == 'mix':
                self.pred_up_dim = nn.Linear(self.edge_repr_dim, self.pooling_dim)
                self.mix_ctx = nn.Linear(self.pooling_dim * 2, self.edge_repr_dim)   


        # final classification
        # change
        self.rel_visual_clf = nn.Linear(self.pooling_dim, self.num_rel_cls)
        # self.rel_visual_clf = nn.Linear(1024, self.num_rel_cls)

        rel_final_dim = self.edge_repr_dim
        self.rel_clf = nn.Linear(rel_final_dim, self.num_rel_cls)

        ### change 
        # if Method_flag == "OBB":
        #      self.post_rel2ctx = nn.Linear(rel_final_dim, 1024)
        # else:
        self.post_rel2ctx = nn.Linear(rel_final_dim, self.pooling_dim)
        #####
        # self.post_rel2ctx = nn.Linear(rel_final_dim, self.pooling_dim)
        layer_init(self.rel_visual_clf, xavier=True)
        layer_init(self.rel_clf, xavier=True)
        layer_init(self.post_rel2ctx, xavier=True)

        # about visual feature of union boxes
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # bias module
        self.bias_module = build_bias_module(config, statistics)
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binaries, roi_features, union_features, logger=None):
        """
        Predicate rel label
        Args:
            proposals: objs
            rel_pair_idxs: object pair index of rel to be predicated
            rel_labels: ground truth rel label
            rel_binaries: binary matrix of relationship for each image
            roi_features: visual feature of objs
            union_features: visual feature of the union boxes of obj pairs
            logger: Logger tool

        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        add_losses = {}
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)
        rel_label_gt = torch.cat(rel_labels, dim=0) if rel_labels is not None else None

        # ### context
        obj_dists, obj_preds, obj_feats, ebd_vector = self.context_layer(roi_features, proposals, logger=logger)
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        # rel encoding
        obj_repr_for_edge = self.post_obj_edge_repr(obj_feats).view(-1, 2, self.hidden_dim)
        edge_rep, obj_pair_labels = self.composeEdgeRepr(obj_repr_for_edge=obj_repr_for_edge, obj_preds=obj_preds,
                                                         rel_pair_idxs=rel_pair_idxs, num_objs=num_objs)

        rel_positive_prob = torch.ones_like(edge_rep[:, 0])

        # graph module
        if self.use_graph_encode:
            if self.use_rel_graph:
                rel_adj_list = self.build_rel_graph(rel_positive_prob, num_rels, rel_pair_idxs, num_objs)
            else:
                rel_adj_list = [None] * len(num_rels)
            # union_features
            if self.graph_enc_strategy == 'cat_gcn':
                pred_rep_list = []
                union_features_down_dim = self.ctx_down_dim(union_features)
                edge_rep = torch.cat((edge_rep, union_features_down_dim), dim=1)
                edge_rep = self.mix_ctx(edge_rep)
                for img_pred_feat, adj in zip(torch.split(edge_rep, num_rels), rel_adj_list):
                    for encoder in self.graph_encoder:
                        img_pred_feat = encoder(img_pred_feat, adj)
                    pred_rep_list.append(img_pred_feat)
                edge_rep = torch.cat(pred_rep_list, dim=0)
            elif self.graph_enc_strategy == 'trans':
                edge_rep = self.pred_up_dim(edge_rep)
                edge_rep = torch.cat((edge_rep, union_features), dim=1)
                edge_rep = self.mix_ctx(edge_rep)
                if not self.use_rel_graph:
                    rel_adj_list = None
                for encoder in self.graph_encoder:
                    edge_rep = encoder(edge_rep, num_rels, rel_adj_list)
            elif self.graph_enc_strategy == 'cross_trans':
                edge_rep = self.pred_up_dim(edge_rep)
                edge_rep = torch.cat((edge_rep, union_features), dim=1)
                edge_rep = self.mix_ctx(edge_rep)
                edge_repr_list = edge_rep.split(num_rels)
                obj_repr_list = obj_feats.split(num_objs)
                edge_enc_results = []
                for i in range(len(num_objs)):
                    if num_rels[i] == 0 or num_objs[i] == 0:
                        continue
                    for encoder in self.graph_encoder:
                        edge_enc_results.append(encoder(edge_repr_list[i], obj_repr_list[i], obj_repr_list[i]))
                edge_rep = torch.cat(edge_repr_list, dim=0)
            elif self.graph_enc_strategy == 'all_trans':
                edge_rep = self.pred_up_dim(edge_rep)
                edge_rep = torch.cat((edge_rep, union_features), dim=1)
                edge_rep = self.mix_ctx(edge_rep)
                edge_repr_list = edge_rep.split(num_rels)
                obj_repr_list = obj_feats.split(num_objs)
                edge_enc_results = []
                for i in range(len(num_objs)):
                    if num_rels[i] == 0 or num_objs[i] == 0:
                        continue
                    for encoder in self.graph_encoder_NE:
                        edge_enc_results.append(encoder(edge_repr_list[i], obj_repr_list[i], obj_repr_list[i]))
                edge_rep = torch.cat(edge_repr_list, dim=0)
                if not self.use_rel_graph:
                    rel_adj_list = None
                for encoder in self.graph_encoder_EE:
                    edge_rep = encoder(edge_rep, num_rels, rel_adj_list)
            elif self.graph_enc_strategy == 'mix':
                edge_rep = self.pred_up_dim(edge_rep)
                edge_rep = torch.cat((edge_rep, union_features), dim=1)
                edge_rep = self.mix_ctx(edge_rep)
            else:
                pred_rep_list = []
                if not self.use_rel_graph:
                    rel_adj_list = [None] * len(num_rels)
                for img_pred_feat, adj in zip(torch.split(edge_rep, num_rels), rel_adj_list):
                    for encoder in self.graph_encoder:
                        img_pred_feat = encoder(img_pred_feat, adj)
                    pred_rep_list.append(img_pred_feat)
                edge_rep = torch.cat(pred_rep_list, dim=0)

        # ### rel classification
        rel_dists = self.rel_classification(edge_rep, obj_pair_labels, union_features, obj_preds, rel_pair_idxs,
                                            num_rels, rel_label_gt, proposals)
        # if not self.training:
        #     return obj_dists, rel_dists, add_losses, obj_preds
        # else:
        return obj_dists, rel_dists, add_losses

    def composeEdgeRepr(self, obj_repr_for_edge, obj_preds, rel_pair_idxs, num_objs):
        # from object level feature to pairwise relation level feature
        pred_reps = []
        pair_preds = []

        head_rep = obj_repr_for_edge[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = obj_repr_for_edge[:, 1].contiguous().view(-1, self.hidden_dim)
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            pred_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        pair_rel_rep = cat(pred_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        return pair_rel_rep, pair_pred

    def rel_classification(self, pred_rep, obj_pair_labels, union_features, obj_preds, rel_pair_idxs,
                           num_rels, rel_label_gt, proposals):
        # rel clf
        rel_dists = self.rel_clf(pred_rep)
        # remove bias
        if not self.training and self.cfg.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.REMOVE_BIAS:
            rel_dists = rel_dists - self.rel_clf.bias

        # use union box and mask convolution
        if self.use_vision:
            ctx_gate = self.post_rel2ctx(pred_rep)  ## change
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features
            rel_dists = rel_dists + self.rel_visual_clf(visual_rep)

        # ### use bias module
        bias = self.bias_module(obj_pair_labels=obj_pair_labels, num_rels=num_rels, obj_preds=obj_preds,
                                gt=rel_label_gt, bbox=[proposal.bbox for proposal in proposals],
                                rel_pair_idxs=rel_pair_idxs)
        if bias is not None:
            rel_dists = rel_dists + bias

        # format operation
        rel_dists = rel_dists.split(num_rels, dim=0)
        return rel_dists

    def build_rel_graph(self, rel_positive_prob, num_rels, rel_pair_idxs, num_objs):
        """
        build rel adjust matrix based on rough clf result
        Args:
            rel_positive_prob:
            num_rels:
            rel_pair_idxs:
            num_objs:

        Returns: adj matrix of rels

        """
        positive_rel_split = torch.split(rel_positive_prob, num_rels)
        rel_graph = []
        for rel_cls, rel_pair_idx, num_obj in zip(positive_rel_split, rel_pair_idxs, num_objs):
            num_rel = rel_pair_idx.size(0)

            rel_obj_matrix = torch.zeros((num_rel, num_obj), device=rel_cls.device)
            if self.eval_fc and not self.training:
                #  in test use fc, for debug
                rel_obj_matrix += 1

            idx = torch.arange(num_rel)
            valid_score = rel_cls.float()

            rel_obj_matrix[idx, rel_pair_idx[:, 0]] += valid_score
            rel_obj_matrix[idx, rel_pair_idx[:, 1]] += valid_score

            adj = torch.matmul(rel_obj_matrix, rel_obj_matrix.T)

            adj[idx, idx] = 1

            adj = adj + self.epsilon
            rel_graph.append(adj)

        return rel_graph
