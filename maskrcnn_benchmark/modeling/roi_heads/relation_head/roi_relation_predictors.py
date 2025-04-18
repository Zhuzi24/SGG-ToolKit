# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import numpy as np
import torch
from maskrcnn_benchmark.modeling import registry
from torch import nn
from torch.nn import functional as F
Tensor = torch.Tensor
from .model_motifs import LSTMContext, FrequencyBias

from maskrcnn_benchmark.modeling.utils import cat
from .model_msg_passing import IMPContext

from .model_vctree import VCTreeLSTMContext
from .model_motifs import LSTMContext
from .model_motifs_with_attribute import AttributeLSTMContext
from .model_transformer import TransformerContext
from .modules.bias_module import build_bias_module
from .utils_relation import layer_init, get_box_info, get_box_pair_info,get_box_info_or,get_box_pair_info_or
# from maskrcnn_benchmark.data import get_dataset_statistics
from .model_dual_transformer import DualTransPredictor
# from extra.utils_funcion import FrequencyBias_GCL
import copy
import math
# from maskrcnn_benchmark.modeling.kl_divergence import KL_divergence
import random
from .utils_motifs import encode_orientedbox_info

from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_relation import nms_overlaps_rotated

from .utils_motifs import rel_vectors, obj_edge_vectors, to_onehot, nms_overlaps, encode_box_info 
## for penet 
from .utils_motifs import to_onehot, encode_box_info
from maskrcnn_benchmark.modeling.make_layers import make_fc


from sklearn.cluster import MiniBatchKMeans
from joblib import Parallel, delayed

from .agcn import _GraphConvolutionLayer_Collect, _GraphConvolutionLayer_Update
from sklearn.cluster import KMeans
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.roi_heads.relation_head.model_msg_passing import (
    PairwiseFeatureExtractor,
)
###BGNN
from maskrcnn_benchmark.modeling.roi_heads.relation_head.classifier import build_classifier

from .utils_relation import obj_prediction_nms
from maskrcnn_benchmark.structures.boxlist_ops import squeeze_tensor
### hestgg 
from maskrcnn_benchmark.structures.boxlist_ops import squeeze_tensor
from maskrcnn_benchmark.modeling.roi_heads.relation_head.model_HetSGG import HetSGG

from math import sqrt
import pandas as pd

###MLPrototype
from .model_motifs import LSTMContext, FrequencyBias
from .utils_prototype import *
from .utils_relation import layer_init
from .utils_motifs import obj_edge_vectors, rel_vectors, encode_box_info, nms_overlaps, to_onehot
from .model_Hybrid_Attention import SHA_Context, SHA_Encoder

import torch
import numpy as np


@registry.ROI_RELATION_PREDICTOR.register("HetSGG_Predictor")
class HetSGG_Predictor(nn.Module):
    def __init__(self, config, in_channels):
        super(HetSGG_Predictor, self).__init__()
        self.num_obj_cls = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES # Duplicate
        self.num_rel_cls = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES # Duplicate
        self.use_bias = cfg.MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS
        self.rel_aware_model_on = config.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON
        self.use_obj_recls_logits = config.MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS

        self.rel_aware_loss_eval = None

        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            self.mode = "predcls" if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL else "sgcls"
        else:
            self.mode = "sgdet"
            
        self.obj_recls_logits_update_manner = (
            cfg.MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_MANNER
        )

        self.n_reltypes = cfg.MODEL.ROI_RELATION_HEAD.HETSGG.NUM_RELATION
        self.n_dim = cfg.MODEL.ROI_RELATION_HEAD.HETSGG.H_DIM
        self.n_ntypes = int(sqrt(self.n_reltypes))
        self.obj2rtype = {(i, j): self.n_ntypes*j+i for j in range(self.n_ntypes) for i in range(self.n_ntypes)}

        self.rel_classifier = build_classifier(self.num_rel_cls, self.num_rel_cls) # Linear Layer
        self.obj_classifier = build_classifier(self.num_obj_cls, self.num_obj_cls)

        self.context_layer = HetSGG(config, in_channels)

        assert in_channels is not None
        if self.use_bias:
            from maskrcnn_benchmark.data import get_dataset_statistics
            statistics = get_dataset_statistics(config)
            self.freq_bias = FrequencyBias(config, statistics)
            self.freq_lambda = nn.Parameter(
                torch.Tensor([1.0]), requires_grad=False
            )  # hurt performance when set learnable


    def init_classifier_weight(self):
        self.obj_classifier.reset_parameters()
        for i in self.n_reltypes:
            self.rel_classifier[i].reset_parameters()


    def forward(
        self,
        inst_proposals,
        rel_pair_idxs,
        rel_labels,
        rel_binarys,
        roi_features,
        union_features,
        logger=None,
        is_training=True,
    ):
        obj_feats, rel_feats = self.context_layer(roi_features, union_features, inst_proposals, rel_pair_idxs, rel_binarys, logger, is_training) # GNN
    
        if self.mode == "predcls":
            obj_labels = cat([proposal.get_field("labels") for proposal in inst_proposals], dim=0)
            refined_obj_logits = to_onehot(obj_labels, self.num_obj_cls)
        else:
            refined_obj_logits = self.obj_classifier(obj_feats)

        rel_cls_logits = self.rel_classifier(rel_feats)

        num_objs = [len(b) for b in inst_proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)
        obj_pred_logits = cat([each_prop.get_field("predict_logits") for each_prop in inst_proposals], dim=0)

        if self.use_obj_recls_logits:
            boxes_per_cls = cat([proposal.get_field("boxes_per_cls") for proposal in inst_proposals], dim=0)  # comes from post process of box_head
            # here we use the logits refinements by adding
            if self.obj_recls_logits_update_manner == "add":
                obj_pred_logits = refined_obj_logits + obj_pred_logits
            if self.obj_recls_logits_update_manner == "replace":
                obj_pred_logits = refined_obj_logits
            refined_obj_pred_labels = obj_prediction_nms(
                boxes_per_cls, obj_pred_logits, nms_thresh=0.5
            )
            obj_pred_labels = refined_obj_pred_labels
        else:
            obj_pred_labels = cat([each_prop.get_field("pred_labels") for each_prop in inst_proposals], dim=0)
        
        if self.use_bias:
            obj_pred_labels = obj_pred_labels.split(num_objs, dim=0)
            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_pred_labels):
                pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
            pair_pred = cat(pair_preds, dim=0)
            rel_cls_logits = rel_cls_logits + self.freq_bias.index_with_labels(pair_pred.long())

        obj_pred_logits = obj_pred_logits.split(num_objs, dim=0)
        rel_cls_logits = rel_cls_logits.split(num_rels, dim=0)
        add_losses = {}

        return obj_pred_logits, rel_cls_logits, add_losses



def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)



    
@registry.ROI_RELATION_PREDICTOR.register("RPCM")
class RPCM(nn.Module):
    def __init__(self, config, in_channels):
        super(RPCM, self).__init__()

        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.cfg = config

        assert in_channels is not None
        self.in_channels = in_channels
        self.obj_dim = in_channels
        

        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        from maskrcnn_benchmark.data import get_dataset_statistics
        statistics = get_dataset_statistics(config)

        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        # assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)
        
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM 
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        self.mlp_dim = 2048 # config.MODEL.ROI_RELATION_HEAD.PENET_MLP_DIM
        self.post_emb = nn.Linear(self.obj_dim, self.mlp_dim * 2)  

        self.embed_dim = 300 # config.MODEL.ROI_RELATION_HEAD.PENET_EMBED_DIM
        dropout_p = 0.2 # config.MODEL.ROI_RELATION_HEAD.PENET_DROPOUT
        
        self.pairwise_feature_extractor = PairwiseFeatureExtractor(cfg, in_channels)
        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)  # load Glove for objects
        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=config.GLOVE_DIR, wv_dim=self.embed_dim)   # load Glove for predicates
        self.obj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)
       
        self.W_sub = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_obj = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_pred = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)

        self.gate_sub = nn.Linear(self.mlp_dim*2, self.mlp_dim)  
        self.gate_obj = nn.Linear(self.mlp_dim*2, self.mlp_dim)
        self.gate_pred = nn.Linear(self.mlp_dim*2, self.mlp_dim)

        self.vis2sem = nn.Sequential(*[
            nn.Linear(self.mlp_dim, self.mlp_dim*2), nn.ReLU(True),
            nn.Dropout(dropout_p), nn.Linear(self.mlp_dim*2, self.mlp_dim)
        ])

        self.project_head = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim*2, 2)
        self.project_head2 = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim*2, 2)

        self.linear_sub = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_obj = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_pred = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_rel_rep = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_rel_rep2 = nn.Linear(self.mlp_dim, self.mlp_dim)
        
        self.norm_sub = nn.LayerNorm(self.mlp_dim)
        self.norm_obj = nn.LayerNorm(self.mlp_dim)
        self.norm_rel_rep = nn.LayerNorm(self.mlp_dim)
        self.norm_rel_rep2 = nn.LayerNorm(self.mlp_dim)

        self.dropout_sub = nn.Dropout(dropout_p)
        self.dropout_obj = nn.Dropout(dropout_p)
        self.dropout_rel_rep = nn.Dropout(dropout_p)
        self.dropout_rel_rep2 = nn.Dropout(dropout_p)
        
        self.dropout_rel = nn.Dropout(dropout_p)
        self.dropout_rel2 = nn.Dropout(dropout_p)
        self.dropout_pred = nn.Dropout(dropout_p)
       
        self.down_samp = MLP(self.pooling_dim, self.mlp_dim, self.mlp_dim, 2) 

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        ##### refine object labels
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum= 0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])

        self.type = self.cfg.Type
  
        
        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.obj_dim = in_channels
        self.out_obj = make_fc(self.hidden_dim, self.num_obj_classes) 
        self.lin_obj_cyx = make_fc(self.obj_dim + self.embed_dim + 128, self.hidden_dim)

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'
        
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        self.bias_module = build_bias_module(config, statistics)
        
    
        #####################################################################
        self.Par = config.EXP_nums 
        self.feat_update_step = config.feat_update_step

        if self.feat_update_step > 0:
            self.gcn_collect_feat = _GraphConvolutionLayer_Collect(self.mlp_dim*2, self.mlp_dim*2)
            self.gcn_update_feat = _GraphConvolutionLayer_Update(self.mlp_dim*2, self.mlp_dim*2)
        


    def build_dict(self,lists):
        dict_ = {}
        for i, sublist in enumerate(lists):
            for item in sublist:
                dict_[item] = i
        return dict_

    def construct_edge_data(self,obj_obj_map):
        # 提取非零元素的位置
        edge_indices = obj_obj_map.nonzero(as_tuple=True)
        edge_index = torch.stack(edge_indices, dim=0)
        
        # 创建 edge_value，所有元素都为 1
        edge_value = torch.ones(edge_index.shape[1], dtype=torch.float).cuda()
        edge_value = edge_value / edge_value.sum()
        
        # 构造 A
        A = [(edge_index, edge_value)]
        
        # 打印 A 的形状
        # for j, (edge_index, edge_value) in enumerate(A):
        #     print(f"Edge type {j}:")
        #     print("Edge index shape:", edge_index.shape)
        #     print("Edge value shape:", edge_value.shape)
        
        return A

    def _get_map_idxs(self, proposals, proposal_pairs):
        rel_inds = []
        obj_num = sum([len(proposal) for proposal in proposals])
        rel_num = sum([len(proposal_pair) for proposal_pair in proposal_pairs])
        obj_obj_map = torch.zeros((obj_num, obj_num))
        pred_pred_map = torch.zeros((rel_num, rel_num)).cuda()

        offset = 0
        for proposal, proposal_pair in zip(proposals, proposal_pairs):
            rel_ind_i = proposal_pair
            obj_obj_map_i = torch.eye(len(proposal)).eq(0).float()
            obj_obj_map[offset:offset + len(proposal), offset:offset + len(proposal)] = obj_obj_map_i
            rel_ind_i += offset
            offset += len(proposal)
            rel_inds.append(rel_ind_i)

        rel_inds = torch.cat(rel_inds, 0)

        subj_pred_map = torch.zeros((obj_num, rel_inds.shape[0])).cuda()
        obj_pred_map = torch.zeros((obj_num, rel_inds.shape[0])).cuda()

        subj_pred_map.scatter_(0, rel_inds[:, 0].contiguous().view(1, -1), 1)
        obj_pred_map.scatter_(0, rel_inds[:, 1].contiguous().view(1, -1), 1)

        obj_obj_map = obj_obj_map.type_as(obj_pred_map)
        pred_pred_map = pred_pred_map.type_as(obj_pred_map)

        rel_inds_0 = rel_inds[:, 0]
        rel_inds_1 = rel_inds[:, 1]

        mask = (rel_inds_0.view(-1, 1) == rel_inds_0.view(1, -1)) | (rel_inds_1.view(-1, 1) == rel_inds_1.view(1, -1)) | (rel_inds_0.view(-1, 1) == rel_inds_1.view(1, -1)) | (rel_inds_1.view(-1, 1) == rel_inds_0.view(1, -1))

        pred_pred_map[mask] = 1
        pred_pred_map = pred_pred_map.triu() + pred_pred_map.triu(1).t()
        pred_pred_map.fill_diagonal_(0)

        return rel_inds, obj_obj_map, subj_pred_map, obj_pred_map, pred_pred_map

    def cluster_features(self,features, n_clusters):
        # 
        features_np = features.clone().detach().cpu().numpy()

        # 
        kmeans = KMeans(n_clusters=n_clusters, n_init=10,random_state=0)

        # 
        cluster_indices = kmeans.fit_predict(features_np)

        # 
        centers = torch.from_numpy(kmeans.cluster_centers_)

        #      
        subclusters = [[] for _ in range(n_clusters)]

        # 
        for i, cluster_index in enumerate(cluster_indices):
            subclusters[cluster_index].append(i+1)

        return centers.cuda(), subclusters
    
    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None ):

        add_losses = {}


        augment_obj_feat, rel_feats = self.pairwise_feature_extractor(
        roi_features,
        union_features,
        proposals,
        rel_pair_idxs
        )

        proposal_pairs= copy.deepcopy(rel_pair_idxs)
        
        rel_inds, obj_obj_map, subj_pred_map, obj_pred_map, pred_pred_map = self._get_map_idxs(proposals, proposal_pairs)  # len(proposal_pairs) 2 

        x_obj =  augment_obj_feat 
        x_pred =  rel_feats 
        obj_feats = [x_obj] 
        pred_feats = [x_pred]
        

        for t in range(self.feat_update_step):
            # message from other objects
            source_obj = self.gcn_collect_feat(obj_feats[t], obj_feats[t], obj_obj_map, 4)

            source_rel_sub = self.gcn_collect_feat(obj_feats[t], pred_feats[t], subj_pred_map, 0)
            source_rel_obj = self.gcn_collect_feat(obj_feats[t], pred_feats[t], obj_pred_map, 1)
            source2obj_all = (source_obj + source_rel_sub + source_rel_obj) / 3

            obj_feats.append(self.gcn_update_feat(obj_feats[t], source2obj_all, 0))
            #update predicate logits

            ##### 
            source_sub_rel = self.gcn_collect_feat(pred_feats[t], obj_feats[t], subj_pred_map.t(), 2)
            source_obj_rel = self.gcn_collect_feat(pred_feats[t], obj_feats[t], obj_pred_map.t(), 3)
            source_rel_rel = self.gcn_collect_feat(pred_feats[t], pred_feats[t], pred_pred_map, 5)
            source2rel_all = (source_sub_rel + source_obj_rel + source_rel_rel) / 3
            pred_feats.append(self.gcn_update_feat(pred_feats[t], source2rel_all, 1))  



        roi_features = obj_feats[-1]
        union_features = pred_feats[-1] 
        
        
        
        # refine object labels
        entity_dists, entity_preds, obj_labels = self.refine_obj_labels(roi_features, proposals)
        ##### 

        entity_rep = self.post_emb(roi_features)   # using the roi features obtained from the faster rcnn
        entity_rep = entity_rep.view(entity_rep.size(0), 2, self.mlp_dim)

        sub_rep = entity_rep[:, 1].contiguous().view(-1, self.mlp_dim)    # xs
        obj_rep = entity_rep[:, 0].contiguous().view(-1, self.mlp_dim)    # xo

        entity_embeds = self.obj_embed(entity_preds) # obtaining the word embedding of entities with GloVe 

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        sub_reps = sub_rep.split(num_objs, dim=0)
        obj_reps = obj_rep.split(num_objs, dim=0)
        entity_preds = entity_preds.split(num_objs, dim=0)
        entity_embeds = entity_embeds.split(num_objs, dim=0)

        fusion_so = []
        pair_preds = []

        for pair_idx, sub_rep, obj_rep, entity_pred, entity_embed, proposal in zip(rel_pair_idxs, sub_reps, obj_reps, entity_preds, entity_embeds, proposals):

            s_embed = self.W_sub(entity_embed[pair_idx[:, 0]])  #  Ws x ts
            o_embed = self.W_obj(entity_embed[pair_idx[:, 1]])  #  Wo x to

            sem_sub = self.vis2sem(sub_rep[pair_idx[:, 0]])  # h(xs)
            sem_obj = self.vis2sem(obj_rep[pair_idx[:, 1]])  # h(xo)
            
            gate_sem_sub = torch.sigmoid(self.gate_sub(cat((s_embed, sem_sub), dim=-1)))  # gs
            gate_sem_obj = torch.sigmoid(self.gate_obj(cat((o_embed, sem_obj), dim=-1)))  # go

            sub = s_embed + sem_sub * gate_sem_sub  # s = Ws x ts + gs · h(xs)  i.e., s = Ws x ts + vs
            obj = o_embed + sem_obj * gate_sem_obj  # o = Wo x to + go · h(xo)  i.e., o = Wo x to + vo

            ##### for the model convergence
            sub = self.norm_sub(self.dropout_sub(torch.relu(self.linear_sub(sub))) + sub)
            obj = self.norm_obj(self.dropout_obj(torch.relu(self.linear_obj(obj))) + obj)
            #####

            fusion_so.append(fusion_func(sub, obj)) # F(s, o)
            pair_preds.append(torch.stack((entity_pred[pair_idx[:, 0]], entity_pred[pair_idx[:, 1]]), dim=1))

        fusion_so = cat(fusion_so, dim=0)  
        pair_pred = cat(pair_preds, dim=0) 
        ###############################
        sem_pred = self.vis2sem(self.down_samp(union_features))  # h(xu)
        gate_sem_pred = torch.sigmoid(self.gate_pred(cat((fusion_so, sem_pred), dim=-1)))  # gp
        rel_rep1 = fusion_so - sem_pred * gate_sem_pred  #  F(s,o) - gp · h(xu)   i.e., r = F(s,o) - up

        
        predicate_proto = self.W_pred(self.rel_embed.weight)  # c = Wp x tp  i.e., semantic prototypes

               
        predicate_proto1 = predicate_proto
        predicate_proto_np = predicate_proto.detach().cpu().numpy()
        background_class = predicate_proto_np[0]
        other_classes = predicate_proto_np[1:]

        kmeans = KMeans(n_clusters= self.Par-1, n_init=10, random_state=0).fit(other_classes)
        labels = kmeans.labels_

        new_predicates = np.vstack((background_class, kmeans.cluster_centers_))
        predicate_proto2 = torch.Tensor(new_predicates).cuda()
        rel_rep1 = self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep1))) + rel_rep1)
        rel_rep1 = self.project_head(self.dropout_rel(torch.relu(rel_rep1)))   
   
        predicate_proto1 = self.project_head(self.dropout_pred(torch.relu(predicate_proto1)))
        predicate_proto2 = self.project_head(self.dropout_pred(torch.relu(predicate_proto2)))

        rel_rep_norm1 = rel_rep1 / rel_rep1.norm(dim=1, keepdim=True)  # r_norm
        predicate_proto_norm1 = predicate_proto1 / predicate_proto1.norm(dim=1, keepdim=True)  # c_norm
        predicate_proto_norm2 = predicate_proto2 / predicate_proto2.norm(dim=1, keepdim=True)
        rel_dists = rel_rep_norm1 @ predicate_proto_norm1.t() * self.logit_scale.exp()
        ### (Prototype-based Learning  ---- cosine similarity) & (Relation Prediction)

        # the rel_dists will be used to calculate the Le_sim with the ce_loss
        entity_dists = entity_dists.split(num_objs, dim=0)
        
        
        if self.training :


            target_rpredicate_proto_norm1 = predicate_proto_norm1.clone().detach()
            target_rpredicate_proto_norm2 = predicate_proto_norm2.clone().detach()

            simil_mat1 = predicate_proto_norm1 @ target_rpredicate_proto_norm1.t()  
            simil_mat2 = predicate_proto_norm2 @ target_rpredicate_proto_norm2.t() # Semantic Matrix S = C_norm @ C_norm.T


            l21_1 = torch.norm(torch.norm(simil_mat1, p=2, dim=1), p=1) / (59*59)   

            l21_2 = torch.norm(torch.norm(simil_mat2, p=2, dim=1), p=1) / ( self.Par * self.Par ) 
            
            add_losses.update({"l21_1_loss": l21_1})  # Le_sim = ||S||_{2,1}
            add_losses.update({"l21_2_loss": l21_2})


            ### end
            
            ### Prototype Regularization  ---- Euclidean distance 51 --31 1124
            gamma2 = 7.0
            predicate_proto_a1 = predicate_proto1.unsqueeze(dim=1).expand(-1, 59, -1) 
            predicate_proto_b1 = predicate_proto1.detach().unsqueeze(dim=0).expand(59, -1, -1)
            predicate_proto_a2 = predicate_proto2.unsqueeze(dim=1).expand(-1, self.Par, -1) 
            predicate_proto_b2 = predicate_proto2.detach().unsqueeze(dim=0).expand( self.Par , -1, -1)
            proto_dis_mat1 = (predicate_proto_a1 - predicate_proto_b1).norm(dim=2) ** 2# Distance Matrix D, dij = ||ci - cj||_2^2
            proto_dis_mat2 = (predicate_proto_a2 - predicate_proto_b2).norm(dim=2) ** 2  
            
            sorted_proto_dis_mat1, _ = torch.sort(proto_dis_mat1, dim=1)
            sorted_proto_dis_mat2, _ = torch.sort(proto_dis_mat2, dim=1)
            topK_proto_dis1 = sorted_proto_dis_mat1[:, :2].sum(dim=1) / 1   # obtain d-, where k2 = 1
            topK_proto_dis2 = sorted_proto_dis_mat2[:, :2].sum(dim=1) / 1
            dist_loss_1 = torch.max(torch.zeros(59).cuda(), -topK_proto_dis1 + gamma2).mean() # Lr_euc = max(0, -(d-) + gamma2)
            dist_loss_2 = torch.max(torch.zeros( self.Par ).cuda(), -topK_proto_dis2 + gamma2).mean()
            add_losses.update({"dist_loss2_1": dist_loss_1})
            add_losses.update({"dist_loss2_2": dist_loss_2})
            ### end
            
            ###  Prototype-based Learning  ---- Euclidean distance
            rel_labels = cat(rel_labels, dim=0)
            gamma1 = 1.0
            rel_rep_expand = rel_rep1.unsqueeze(dim=1).expand(-1, 59, -1)  # r
            predicate_proto_expand = predicate_proto1.unsqueeze(dim=0).expand(rel_labels.size(0), -1, -1)  # ci
            distance_set = (rel_rep_expand - predicate_proto_expand).norm(dim=2) ** 2    # Distance Set G, gi = ||r-ci||_2^2
            mask_neg = torch.ones(rel_labels.size(0), 59).cuda()  
            mask_neg[torch.arange(rel_labels.size(0)), rel_labels] = 0
            distance_set_neg = distance_set * mask_neg
            distance_set_pos = distance_set[torch.arange(rel_labels.size(0)), rel_labels]  # gt i.e., g+;keyi jiezhe posdistance  jisuan density
            sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
            #K=max(int(0.1*len(sorted_distance_set_neg)),10)
            topK_sorted_distance_set_neg = sorted_distance_set_neg[:, :11].sum(dim=1) / 10  # obtaining g-, where k1 = 10, 
            #topK_sorted_distance_set_neg = sorted_distance_set_neg[:, :6].sum(dim=1) / 5  # obtaining g-, where k1 = 10, 
            loss_sum = torch.max(torch.zeros(rel_labels.size(0)).cuda(), distance_set_pos - topK_sorted_distance_set_neg + gamma1).mean()
            add_losses.update({"loss_dis": loss_sum})     # Le_euc = max(0, (g+) - (g-) + gamma1)


        rel_dists = rel_dists.split(num_rels, dim=0)
        return entity_dists, rel_dists, add_losses  
                



    def refine_obj_labels(self, roi_features, proposals):
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None
        
        if  proposals[0].bbox.shape[-1] == 5: 
            pos_embed = self.pos_embed(encode_orientedbox_info(proposals))
        else:    
            pos_embed = self.pos_embed(encode_box_info(proposals))

        # label/logits embedding will be used as input
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed1(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight
            # obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight
        
        if  proposals[0].bbox.shape[-1] == 5: 
            assert proposals[0].mode == 'xywha'
        else:
            assert proposals[0].mode == 'xyxy'
        
        if  proposals[0].bbox.shape[-1] == 5: 
            pos_embed = self.pos_embed(encode_orientedbox_info(proposals))
        else:    
            pos_embed = self.pos_embed(encode_box_info(proposals))
        num_objs = [len(p) for p in proposals]
        obj_pre_rep_for_pred = self.lin_obj_cyx(cat([roi_features, obj_embed, pos_embed], -1))

        if self.mode == 'predcls':
            obj_labels = obj_labels.long()
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_classes)
        else:
            obj_dists = self.out_obj(obj_pre_rep_for_pred)  # 512 -> 151
            use_decoder_nms = self.mode == 'sgdet' and not self.training ### change
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs).long()
            else:
                 obj_preds = (obj_dists[:, 1:].max(1)[1] + 1).long()
        
        return obj_dists, obj_preds, obj_labels

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            if ("HBB" in self.type) or ("CV" in self.type) :
               is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh # (#box, #box, #class)
            else:
               is_overlap = nms_overlaps_rotated(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh # (#box, #box, #class)
            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind,:,cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0 # This way we won't re-sample

            obj_preds.append(out_label.long())
        obj_preds = torch.cat(obj_preds, dim=0)
        return obj_preds


    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)  
        return x
    
    
def fusion_func(x, y):
    return F.relu(x + y) - (x - y) ** 2




@registry.ROI_RELATION_PREDICTOR.register("PrototypeEmbeddingNetwork")
class PrototypeEmbeddingNetwork(nn.Module):
    def __init__(self, config, in_channels):
        super(PrototypeEmbeddingNetwork, self).__init__()

        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.cfg = config

        assert in_channels is not None
        self.in_channels = in_channels
        self.obj_dim = in_channels # --0206
        
       
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        from maskrcnn_benchmark.data import get_dataset_statistics
        statistics = get_dataset_statistics(config)

        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        # assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.num_obj_classes = len(obj_classes)
        
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM 
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

         #change for RS
       # self.mlp_dim = 2048 # config.MODEL.ROI_RELATION_HEAD.PENET_MLP_DIM
        self.mlp_dim = 2048

    

        ## change for RS
        # self.post_emb = nn.Linear(1024, self.mlp_dim * 2)  
        self.post_emb = nn.Linear(self.obj_dim, self.mlp_dim * 2)  

        self.embed_dim = 300 # config.MODEL.ROI_RELATION_HEAD.PENET_EMBED_DIM
        dropout_p = 0.2 # config.MODEL.ROI_RELATION_HEAD.PENET_DROPOUT
        
        
        obj_embed_vecs = obj_edge_vectors(obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)  # load Glove for objects
        rel_embed_vecs = rel_vectors(rel_classes, wv_dir=config.GLOVE_DIR, wv_dim=self.embed_dim)   # load Glove for predicates
        self.obj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)
        self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)
       
        self.W_sub = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_obj = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_pred = MLP(self.embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)

        self.gate_sub = nn.Linear(self.mlp_dim*2, self.mlp_dim)  
        self.gate_obj = nn.Linear(self.mlp_dim*2, self.mlp_dim)
        self.gate_pred = nn.Linear(self.mlp_dim*2, self.mlp_dim)

        ### for RS change 
        # self.vis2sem = nn.Sequential(*[
        #     nn.Linear(512, 512*2), nn.ReLU(True),
        #     nn.Dropout(dropout_p), nn.Linear(512*2, 512)
        # ])
        

        self.vis2sem = nn.Sequential(*[
            nn.Linear(self.mlp_dim, self.mlp_dim*2), nn.ReLU(True),
            nn.Dropout(dropout_p), nn.Linear(self.mlp_dim*2, self.mlp_dim)
        ])

        self.project_head = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim*2, 2)

        self.linear_sub = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_obj = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_pred = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_rel_rep = nn.Linear(self.mlp_dim, self.mlp_dim)
        
        self.norm_sub = nn.LayerNorm(self.mlp_dim)
        self.norm_obj = nn.LayerNorm(self.mlp_dim)
        self.norm_rel_rep = nn.LayerNorm(self.mlp_dim)

        self.dropout_sub = nn.Dropout(dropout_p)
        self.dropout_obj = nn.Dropout(dropout_p)
        self.dropout_rel_rep = nn.Dropout(dropout_p)
        
        self.dropout_rel = nn.Dropout(dropout_p)
        self.dropout_pred = nn.Dropout(dropout_p)
       
        # self.down_samp = MLP(self.pooling_dim, self.mlp_dim, self.mlp_dim, 2) ### for RS change 
        ## 原始 4096 2048 2048 
        ## now 1024 512 512
        # self.down_samp = MLP(1024, 2048, 2048, 2)
        self.down_samp = MLP(self.pooling_dim, self.mlp_dim, self.mlp_dim, 2)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        ##### refine object labels
        self.pos_embed = nn.Sequential(*[
            nn.Linear(9, 32), nn.BatchNorm1d(32, momentum= 0.001),
            nn.Linear(32, 128), nn.ReLU(inplace=True),
        ])

        self.obj_embed1 = nn.Embedding(self.num_obj_classes, self.embed_dim)
        with torch.no_grad():
            self.obj_embed1.weight.copy_(obj_embed_vecs, non_blocking=True)
  
        #### change for RS
        # self.obj_dim = in_channels
        self.obj_dim = in_channels
        self.out_obj = make_fc(self.hidden_dim, self.num_obj_classes) 
        self.lin_obj_cyx = make_fc(self.obj_dim + self.embed_dim + 128, self.hidden_dim)

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'
        
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        self.par = 16

        self.allids = [x for x  in range(31)]

        self.type = self.cfg.Type

        self.wei = torch.FloatTensor([0.0003046222352368687, 0.007538628316553855, 0.22779721097623237, 0.008090328589242385, 0.004788360528431013, 0.008722856092222073, 0.009663001582475697, 0.11807846471412811, 0.01053165478265624, 0.018837946971245106, 0.04492701436365913, 0.006997075904319194, 0.0029003808471346655, 0.11502677471433329, 0.08842945608466789, 0.14367903563700554, 0.03163535734172965, 0.07318067598175781, 0.020896024488812486, 0.01328348283360746, 1.1510235842780039, 0.005824025923631396, 0.006198435020185267, 0.0022211519655884627, 0.0398630798984748, 0.25074527902371646, 0.25695832630312915, 0.005851050831113645, 0.8514817501713848, 0.029902235188948762, 0.05010990030250687, 0.035066604728872096, 0.24676761385401302, 0.00494885069738179, 0.03184406373426899, 0.08436249561260968, 0.05130229636277002, 0.014897093799680218, 0.012932486102513501, 1.0189563906393357, 0.07431357098312755, 31.073676505191273, 0.8178766414667756, 0.6984337647517445, 0.8514817501713848, 0.3062954318832857, 0.18133908379043306, 0.07793344731709655, 0.19497068431463968, 5.179073011596186, 0.06002435587003362, 0.017017640297374723, 0.08268500559325792, 1.5936663771966852, 1.8280066796961938, 0.23555783028442892, 0.09035130597518326, 1.1727381363507927, 10.357993709846525]).cuda()


    # def cluster_features(self, features, n_clusters):
    #     # 将输入张量转换为numpy数组
    #     features_np = features.clone().detach().cpu().numpy()

    #     # 创建KMeans对象
    #     kmeans = KMeans(n_clusters=n_clusters, random_state=0)

    #     # 拟合数据并预测每个数据点的簇索引
    #     kmeans.fit(features_np)

    #     # 获取簇中心并转换为张量
    #     centers = torch.from_numpy(kmeans.cluster_centers_).cuda()

    #     return centers

    def cluster_features(self,features, n_clusters):
        # 将输入张量转换为numpy数组
        features_np = features.clone().detach().cpu().numpy()

        # 创建KMeans对象
        kmeans = KMeans(n_clusters=n_clusters, n_init=10,random_state=0)

        # 拟合数据并预测每个数据点的簇索引
        cluster_indices = kmeans.fit_predict(features_np)

        # 获取簇中心并转换为张量
        centers = torch.from_numpy(kmeans.cluster_centers_)

        # 创建一个空列表来保存每个簇的子类索引
        subclusters = [[] for _ in range(n_clusters)]

        # 遍历每个数据点，将其添加到对应的簇索引列表中
        for i, cluster_index in enumerate(cluster_indices):
            subclusters[cluster_index].append(i+1)

        return centers.cuda(), subclusters

    # def cluster_and_min(self,tensor):
    #     # 创建一个用于存储结果的张量
    #     result = torch.zeros(tensor.shape[0])
        
    #     # 对张量的每一行进行处理
    #     for i in range(tensor.shape[0]):
    #         # 获取当前行
    #         row = tensor[i, :].clone().detach().cpu().numpy()
    #         row = row[row != 0].reshape(-1, 1)
                        
    #         # 执行KMeans聚类
    #         kmeans = KMeans(n_clusters=2, random_state=0,n_init=10).fit(row)
            
    #         # 计算每个簇的平均值
    #         cluster_averages = [np.mean(row[kmeans.labels_ == j]) for j in range(2)]
            
    #         # 获取平均值最小的簇的平均值
    #         min_average = min(cluster_averages)
            
    #         # 将结果存储在结果张量中
    #         result[i] = torch.tensor(min_average)
        
    #     return result.cuda()



    # def cluster_and_min(self,tensor):
    #     # 创建一个用于存储结果的张量
    #     result = torch.zeros(tensor.shape[0])

    #     def process_row(i):
    #         # 获取当前行
    #         row = tensor[i, :].clone().detach().cpu().numpy()
    #         row = row[row != 0].reshape(-1, 1)
                        
    #         # 执行MiniBatchKMeans聚类
    #         kmeans = MiniBatchKMeans(n_clusters=2, random_state=0).fit(row)
            
    #         # 计算每个簇的平均值
    #         cluster_averages = [np.mean(row[kmeans.labels_ == j]) for j in range(2)]
            
    #         # 获取平均值最小的簇的平均值
    #         min_average = min(cluster_averages)
            
    #         return min_average

    #     # 使用joblib并行处理每一行
    #     result[:] = torch.tensor(Parallel(n_jobs=-1)(delayed(process_row)(i) for i in range(tensor.shape[0])))

    #     return result.cuda()


    def build_dict(self,lists):
        dict_ = {}
        for i, sublist in enumerate(lists):
            for item in sublist:
                dict_[item] = i
        return dict_

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):

        add_losses = {}
        add_data = {}

        # refine object labels
        entity_dists, entity_preds = self.refine_obj_labels(roi_features, proposals)
        ##### 

        entity_rep = self.post_emb(roi_features)   # using the roi features obtained from the faster rcnn   ## 已经change
        entity_rep = entity_rep.view(entity_rep.size(0), 2, self.mlp_dim)

        sub_rep = entity_rep[:, 1].contiguous().view(-1, self.mlp_dim)    # xs 视觉特征
        obj_rep = entity_rep[:, 0].contiguous().view(-1, self.mlp_dim)    # xo

        entity_embeds = self.obj_embed(entity_preds) # obtaining the word embedding of entities with GloVe   标签的编码 nums,300

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        sub_reps = sub_rep.split(num_objs, dim=0)
        obj_reps = obj_rep.split(num_objs, dim=0)
        entity_preds = entity_preds.split(num_objs, dim=0)
        entity_embeds = entity_embeds.split(num_objs, dim=0)

        fusion_so = []
        pair_preds = []

        for pair_idx, sub_rep, obj_rep, entity_pred, entity_embed, proposal in zip(rel_pair_idxs, sub_reps, obj_reps, entity_preds, entity_embeds, proposals):

            s_embed = self.W_sub(entity_embed[pair_idx[:, 0]])  #  Ws x ts  词嵌入  nums,2048
            o_embed = self.W_obj(entity_embed[pair_idx[:, 1]])  #  Wo x to   nums,2048

            sem_sub = self.vis2sem(sub_rep[pair_idx[:, 0]])  # h(xs)  视觉特征   # nums,2048
            sem_obj = self.vis2sem(obj_rep[pair_idx[:, 1]])  # h(xo)   # nums,2048
            
            gate_sem_sub = torch.sigmoid(self.gate_sub(cat((s_embed, sem_sub), dim=-1)))  # gs  0-1 
            gate_sem_obj = torch.sigmoid(self.gate_obj(cat((o_embed, sem_obj), dim=-1)))  # go  0-1

            sub = s_embed + sem_sub * gate_sem_sub  # s = Ws x ts + gs · h(xs)  i.e., s = Ws x ts + vs
            obj = o_embed + sem_obj * gate_sem_obj  # o = Wo x to + go · h(xo)  i.e., o = Wo x to + vo

            ##### for the model convergence
            sub = self.norm_sub(self.dropout_sub(torch.relu(self.linear_sub(sub))) + sub)
            obj = self.norm_obj(self.dropout_obj(torch.relu(self.linear_obj(obj))) + obj)
            #####

            fusion_so.append(fusion_func(sub, obj)) # F(s, o)
            pair_preds.append(torch.stack((entity_pred[pair_idx[:, 0]], entity_pred[pair_idx[:, 1]]), dim=1))

        fusion_so = cat(fusion_so, dim=0)  
        pair_pred = cat(pair_preds, dim=0) 

        sem_pred = self.vis2sem(self.down_samp(union_features))  # h(xu)  ## for RS num,1024
        gate_sem_pred = torch.sigmoid(self.gate_pred(cat((fusion_so, sem_pred), dim=-1)))  # gp

        rel_rep = fusion_so - sem_pred * gate_sem_pred  #  F(s,o) - gp · h(xu)   i.e., r = F(s,o) - up
        predicate_proto = self.W_pred(self.rel_embed.weight)  # c = Wp x tp  i.e., semantic prototypes   31 2048
        
        ##### for the model convergence
        rel_rep = self.norm_rel_rep(self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))) + rel_rep)  ### torch.Size([nums, 2048])

        rel_rep = self.project_head(self.dropout_rel(torch.relu(rel_rep))) ### nums,4096
        predicate_proto = self.project_head(self.dropout_pred(torch.relu(predicate_proto))) ## 31,4096
        ######

        rel_rep_norm = rel_rep / rel_rep.norm(dim=1, keepdim=True)  # r_norm
        predicate_proto_norm = predicate_proto / predicate_proto.norm(dim=1, keepdim=True)  # c_norm

        ### (Prototype-based Learning  ---- cosine similarity) & (Relation Prediction)
        rel_dists = rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()  #  <r_norm, c_norm> / τ   ### nums, 31
        # the rel_dists will be used to calculate the Le_sim with the ce_loss

        entity_dists = entity_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)


        if self.training:


            ### Prototype Regularization  ---- cosine similarity
            target_rpredicate_proto_norm = predicate_proto_norm.clone().detach()
            simil_mat = predicate_proto_norm @ target_rpredicate_proto_norm.t()  # Semantic Matrix S = C_norm @ C_norm.T  ## 31.31


            l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (len(self.rel_classes) * len(self.rel_classes))  ##  self.rel_classes * self.rel_classes
            add_losses.update({"l21_loss": l21})  # Le_sim = ||S||_{2,1}
            # ### end


            ### Prototype Regularization  ---- Euclidean distance 51 --31 1124
            gamma2 = 7.0
            predicate_proto_a = predicate_proto.unsqueeze(dim=1).expand(-1, 59, -1)
            predicate_proto_b = predicate_proto.detach().unsqueeze(dim=0).expand(59, -1, -1)
            proto_dis_mat = (predicate_proto_a - predicate_proto_b).norm(dim=2) ** 2  # Distance Matrix D, dij = ||ci - cj||_2^2
            sorted_proto_dis_mat, _ = torch.sort(proto_dis_mat, dim=1)
            topK_proto_dis = sorted_proto_dis_mat[:, :2].sum(dim=1) / 1   # obtain d-, where k2 = 1
            dist_loss = torch.max(torch.zeros(59).cuda(), -topK_proto_dis + gamma2).mean()  # Lr_euc = max(0, -(d-) + gamma2)
            add_losses.update({"dist_loss2": dist_loss})
            ### end

            ###  Prototype-based Learning  ---- Euclidean distance
            rel_labels = cat(rel_labels, dim=0)
            gamma1 = 1.0
            rel_rep_expand = rel_rep.unsqueeze(dim=1).expand(-1, 59, -1)  # r
            predicate_proto_expand = predicate_proto.unsqueeze(dim=0).expand(rel_labels.size(0), -1, -1)  # ci
            distance_set = (rel_rep_expand - predicate_proto_expand).norm(dim=2) ** 2    # Distance Set G, gi = ||r-ci||_2^2   # nums,31
            mask_neg = torch.ones(rel_labels.size(0), 59).cuda()  # nums,31
            mask_neg[torch.arange(rel_labels.size(0)), rel_labels] = 0 ##  torch.arange(rel_labels.size(0)) 0-90,  rel_labels 长为90
            distance_set_neg = distance_set * mask_neg
            distance_set_pos = distance_set[torch.arange(rel_labels.size(0)), rel_labels]  # gt i.e., g+
            sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)  ## dis - 小到大
            topK_sorted_distance_set_neg = sorted_distance_set_neg[:, :11].sum(dim=1) / 10  # obtaining g-, where k1 = 10,
            loss_sum = torch.max(torch.zeros(rel_labels.size(0)).cuda(), distance_set_pos - topK_sorted_distance_set_neg + gamma1).mean()
            add_losses.update({"loss_dis": loss_sum})     # Le_euc = max(0, (g+) - (g-) + gamma1)
            ### end


        return entity_dists, rel_dists, add_losses, add_data






    def refine_obj_labels(self, roi_features, proposals):
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None
        
        if  proposals[0].bbox.shape[-1] == 5: 
            pos_embed = self.pos_embed(encode_orientedbox_info(proposals))
        else:    
            pos_embed = self.pos_embed(encode_box_info(proposals))


        # label/logits embedding will be used as input
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = obj_labels.long()
            obj_embed = self.obj_embed1(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()


            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight
            # obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed1.weight
        
        if  proposals[0].bbox.shape[-1] == 5: 
            assert proposals[0].mode == 'xywha'
        else:
            assert proposals[0].mode == 'xyxy'
        
        if  proposals[0].bbox.shape[-1] == 5: 
            pos_embed = self.pos_embed(encode_orientedbox_info(proposals))
        else:    
            pos_embed = self.pos_embed(encode_box_info(proposals))
        num_objs = [len(p) for p in proposals]
        obj_pre_rep_for_pred = self.lin_obj_cyx(cat([roi_features, obj_embed, pos_embed], -1))

        if self.mode == 'predcls':
            obj_labels = obj_labels.long()
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_classes)
        else:
            obj_dists = self.out_obj(obj_pre_rep_for_pred)  # 512 -> 151
            use_decoder_nms = self.mode == 'sgdet' and not self.training ### change
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs).long()
            else:
                obj_preds = (obj_dists[:, 1:].max(1)[1] + 1).long()
        
        return obj_dists, obj_preds

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            if ("HBB" in self.type) or ("CV" in self.type) :
               is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh # (#box, #box, #class)
            else:
               is_overlap = nms_overlaps_rotated(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh # (#box, #box, #class)
            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind,:,cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0 # This way we won't re-sample

            obj_preds.append(out_label.long())
        obj_preds = torch.cat(obj_preds, dim=0)
        return obj_preds

    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)  
        return x
    
    
def fusion_func(x, y):
    return F.relu(x + y) - (x - y) ** 2






@registry.ROI_RELATION_PREDICTOR.register("TransformerPredictor")
class TransformerPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(TransformerPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

 

        # load class dict
        from maskrcnn_benchmark.data import get_dataset_statistics
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        # assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        # module construct
        if config.BASIC_ENCODER == 'Hybrid-Attention':
            self.context_layer = SHA_Context(config, obj_classes, rel_classes, in_channels)
        else:
            self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # bias module
        self.bias_module = build_bias_module(config, statistics)


    # def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None,
                iteration=None, m=None, val=None, uni_tem=None,GLO_f = None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)

        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features

        rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)

        gt = torch.cat(rel_labels, dim=0) if rel_labels is not None else None
        
        
        if self.use_bias:
            bias = self.bias_module.index_with_labels(pair_pred.long(), gt=gt)
            if bias is not None:
                rel_dists = rel_dists +  bias 

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0) ## [nums,class]


        add_losses = {}

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
                return obj_dists, rel_dists, add_losses #,num_objs,num_rels


@registry.ROI_RELATION_PREDICTOR.register("IMPPredictor")
class IMPPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(IMPPredictor, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.use_bias = False

        assert in_channels is not None

        self.context_layer = IMPContext(config, self.num_obj_cls, self.num_rel_cls, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # freq 
        if self.use_bias:
            from maskrcnn_benchmark.data import get_dataset_statistics
            statistics = get_dataset_statistics(config)
            self.freq_bias = build_bias_module(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        # encode context infomation
        obj_dists, rel_dists = self.context_layer(roi_features, proposals, union_features, rel_pair_idxs, logger)

        num_objs = [len(b) for b in proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)

        if self.use_bias:
            obj_preds = obj_dists.max(-1)[1]
            obj_preds = obj_preds.split(num_objs, dim=0)

            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_preds):
                pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
            pair_pred = cat(pair_preds, dim=0)

            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        return obj_dists, rel_dists, add_losses




@registry.ROI_RELATION_PREDICTOR.register("MotifPredictor")
class MotifPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(MotifPredictor, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS   # PREDICT_USE_BIAS

        # load class dict
        from maskrcnn_benchmark.data import get_dataset_statistics
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        # assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        if self.attribute_on:
            self.context_layer = AttributeLSTMContext(config, obj_classes, att_classes, rel_classes, in_channels)
        else:
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)

        # initialize layer parameters
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        layer_init(self.rel_compress, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # bias module
        self.bias_module = build_bias_module(config, statistics)
        

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None,ite = None,
                m=None, uni_tem=None, bce = None, val = None,vae = None, s1_index = None):

        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, logger)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)

        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:
                prod_rep = prod_rep * union_features

        rel_dists = self.rel_compress(prod_rep)


        gt = torch.cat(rel_labels, dim=0) if rel_labels is not None else None
        if self.use_bias:   
            bias = self.bias_module.index_with_labels(pair_pred.long(), gt=gt)
            if bias is not None:
                rel_dists = rel_dists + bias


        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses



@registry.ROI_RELATION_PREDICTOR.register("VCTreePredictor")
class VCTreePredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(VCTreePredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        from maskrcnn_benchmark.data import get_dataset_statistics
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics[
            'att_classes']
        assert self.num_obj_cls == len(obj_classes)
        # assert self.num_att_cls == len(att_classes)
        assert self.num_rel_cls == len(rel_classes)
        # init contextual lstm encoding
        self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # learned-mixin
        # self.uni_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        # self.frq_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        # self.uni_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        # layer_init(self.uni_gate, xavier=True)
        # layer_init(self.frq_gate, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        # layer_init(self.uni_compress, xavier=True)

        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)

        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # bias module
        self.bias_module = build_bias_module(config, statistics)
        self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger)

        # post decode
        edge_rep = F.relu(self.post_emb(edge_ctx))
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        # learned-mixin Gate
        #uni_gate = torch.tanh(self.uni_gate(self.drop(prod_rep)))
        #frq_gate = torch.tanh(self.frq_gate(self.drop(prod_rep)))

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        ctx_dists = self.ctx_compress(prod_rep * union_features)
        #uni_dists = self.uni_compress(self.drop(union_features))
       
       ## change  frq_dists = self.freq_bias.index_with_labels(pair_pred.long())
        rel_dists = ctx_dists # + frq_dists
        
        #rel_dists = ctx_dists + uni_gate * uni_dists + frq_gate * frq_dists

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.training:
            binary_loss = []
            for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                bi_gt = (bi_gt > 0).float()
                binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
            add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

        return obj_dists, rel_dists, add_losses


