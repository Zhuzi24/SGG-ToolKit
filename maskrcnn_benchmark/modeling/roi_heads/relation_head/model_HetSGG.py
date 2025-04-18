import sys
import json
sys.path.append('')
import numpy as np
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch import Tensor
from torch_scatter import scatter, segment_csr, gather_csr
from torch_scatter import scatter_add


from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_motifs import encode_box_info
from maskrcnn_benchmark.modeling.roi_heads.relation_head.model_msg_passing import PairwiseFeatureExtractor

from maskrcnn_benchmark.modeling.roi_heads.relation_head.classifier import build_classifier




class HetSGG(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(HetSGG, self).__init__()
        
        self.cfg = cfg
        self.n_reltypes = cfg.MODEL.ROI_RELATION_HEAD.HETSGG.NUM_RELATION
        self.n_ntypes = int(sqrt(self.n_reltypes))
        self.num_bases = cfg.MODEL.ROI_RELATION_HEAD.HETSGG.N_BASES
        self.dim = cfg.MODEL.ROI_RELATION_HEAD.HETSGG.H_DIM
        self.score_update_step = cfg.MODEL.ROI_RELATION_HEAD.HETSGG.SCORE_UPDATE_STEP
        self.feat_update_step = cfg.MODEL.ROI_RELATION_HEAD.HETSGG.FEATURE_UPDATE_STEP
        self.geometry_feat_dim = 128
        self.hidden_dim = cfg.MODEL.ROI_RELATION_HEAD.HETSGG.H_DIM
        num_classes_obj = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        num_classes_pred = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        
        self.vg_cat_dict = json.load(open("/media/dell/data1/WTZ/SGG_Frame/data/labels.json", 'r'))

        self.vg_map_arr = self.compute_category_mapping_array()

        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            self.mode = 'predcls' if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL else 'sgcls'
        else:
            self.mode = "sgdet"

        self.obj2rtype = {(i, j): self.n_ntypes*j+i for j in range(self.n_ntypes) for i in range(self.n_ntypes)}

        self.gt2pred = []

        self.pairwise_feature_extractor = PairwiseFeatureExtractor(cfg, in_channels)

        self.rel_embedding = nn.Sequential(
            nn.Linear(self.pairwise_feature_extractor.pooling_dim, self.dim*2),
            nn.ReLU(True),
            nn.Linear(self.dim*2, self.dim),
            nn.ReLU(True) 
        )
        self.obj_embedding = nn.Sequential(
            nn.Linear(self.pairwise_feature_extractor.pooling_dim, self.dim*2),
            nn.ReLU(True),
            nn.Linear(self.dim*2, self.dim),
            nn.ReLU(True)
             
        )

        self.rel_classifier = build_classifier(self.hidden_dim, num_classes_pred)
        self.obj_classifier = build_classifier(self.hidden_dim, num_classes_obj)

        if self.feat_update_step > 0:
            self.rmp = RMPLayer(self.dim, self.dim, self.dim, self.dim, self.num_bases, self.n_reltypes, cfg=cfg)
       
        if self.score_update_step > 0:
            self.rmp_score = RMPLayer(num_classes_obj, num_classes_pred, num_classes_obj, num_classes_pred, num_bases=self.num_bases, num_relations=self.n_reltypes, cfg=cfg)

        self.init_classifier_weight()


    def init_classifier_weight(self):
        self.rel_classifier.reset_parameters()
        self.obj_classifier.reset_parameters()


    def compute_category_mapping_array(self):
        key = list(self.vg_cat_dict['labelidx_to_catidx'].keys())
        value = list(self.vg_cat_dict['labelidx_to_catidx'].values())
        vg_map_arr = np.array([key, value], dtype = int).transpose()
        vg_map_arr = np.array(vg_map_arr[np.argsort(vg_map_arr[:,0])])
        return torch.LongTensor(vg_map_arr).cuda()


    def forward(self,      
        inst_features,
        rel_union_features,
        proposals,
        rel_pair_inds,
        rel_gt_binarys=None,
        logger=None, is_training=True):

        nf, ef = self.pairwise_feature_extractor( 
            inst_features,
            rel_union_features,
            proposals,
            rel_pair_inds,
        )
        
        g_rel, etype_rel, etype_rel_inv = self._get_map_idxs(proposals, rel_pair_inds, is_training)


        edge_type_list = [etype_rel, etype_rel_inv]

        nf = self.obj_embedding(nf) # Node Feature
        ef = self.rel_embedding(ef) # Edge Feature

        for _ in range(self.feat_update_step):
            nf, ef = self.rmp(g_rel, nf, ef, edge_type_list)
            nf = F.elu(nf)
            ef = F.elu(ef)

        # Relationship Classifier
        pred_class_logits = self.rel_classifier(ef)
        obj_class_logits = self.obj_classifier(nf)

        # Logit Layer
        for _ in range(self.score_update_step):
            obj_class_logits, pred_class_logits = self.rmp_score(g_rel, obj_class_logits, pred_class_logits, edge_type_list)
            obj_class_logits = F.elu(obj_class_logits)
            pred_class_logits = F.elu(pred_class_logits)

        return obj_class_logits, pred_class_logits


    def spatial_embedding(self, proposals):
        """
        Compute the Spatial Information for each proposal
        """
        pos_embed = []
        for proposal in proposals:
            pos_embed.append(encode_box_info([proposals,]))
        pos_embed = torch.cat(pos_embed)
        return pos_embed


    def _get_map_idxs(self, proposals, rel_pair_inds, is_training):

        offset = 0
        rel_inds = []

        edge_types = []
        edge_types_inv = []
        
        for proposal, rel_pair_ind in zip(proposals, rel_pair_inds):

            # Generate Graph
            rel_ind_i = rel_pair_ind.detach().clone()
            rel_ind_i += offset
            rel_inds.append(rel_ind_i)

            # Get Node Type for each entity
            if self.mode in ['sgcls', 'sgdet']:
                proposal_category = proposal.extra_fields['category_scores'].max(1)[1].detach()
            else:
                proposal_category = self.vg_map_arr[proposal.extra_fields['labels'].long(), 1]
            

            edge_type = torch.LongTensor([self.obj2rtype[(s.item(),d.item())] for (s, d) in proposal_category[rel_ind_i.detach()-offset]])
            edge_type_inv = torch.LongTensor([self.obj2rtype[(d.item(), s.item())] for (s, d) in proposal_category[rel_ind_i.detach()-offset]])

            edge_types.append(edge_type)
            edge_types_inv.append(edge_type_inv)

            offset += len(proposal)

        rel_inds = torch.cat(rel_inds, 0).T
       
        edge_types = torch.cat(edge_types).cuda()
        edge_types_inv = torch.cat(edge_types_inv).cuda()

        return rel_inds, edge_types, edge_types_inv


class RMPLayer(MessagePassing):

    def __init__(self, node_in_channels, edge_in_channels, node_out_channels, edge_out_channels, num_bases, num_relations, cfg, bias=False):
        super(RMPLayer, self).__init__()

        self.node_in_channels = node_in_channels
        self.edge_in_channels = edge_in_channels
        self.node_out_channels = node_out_channels
        self.edge_out_channels = edge_out_channels
        self.bias = bias

        self.num_relations = num_relations
        self.num_bases = num_bases
        self.n_ntypes = int(sqrt(self.num_relations))

        self.obj2rtype = {(i, j): self.n_ntypes*j+i for j in range(self.n_ntypes) for i in range(self.n_ntypes)}
        self.rtype2obj = dict(zip(self.obj2rtype.values(), self.obj2rtype.keys()))

        # Check Dimension
        self.sub2rel_basis = nn.Parameter(torch.Tensor(num_bases, node_in_channels, edge_out_channels))
        self.sub2rel_att =  nn.Parameter(torch.Tensor(num_relations, num_bases))

        self.entity2rel_attn = nn.Linear(edge_out_channels, 1, bias = False)

        
        self.obj2rel_basis = nn.Parameter(torch.Tensor(num_bases, node_in_channels, edge_out_channels))
        self.obj2rel_att =  nn.Parameter(torch.Tensor(num_relations, num_bases)) 

        self.rel2obj_basis = nn.Parameter(torch.Tensor(num_bases, edge_in_channels, node_out_channels))
        self.rel2obj_att = nn.Parameter(torch.Tensor(num_relations, num_bases))
        self.rel2obj_attn = nn.ModuleList([nn.Linear(node_out_channels, 1) for _ in range(self.num_relations)])
        
        self.rel2sub_basis = nn.Parameter(torch.Tensor(num_bases, edge_in_channels, node_out_channels))
        self.rel2sub_att = nn.Parameter(torch.Tensor(num_relations, num_bases))
        self.rel2sub_attn = nn.ModuleList([nn.Linear(node_out_channels, 1) for _ in range(self.num_relations)])

        self.reset_parameters()


    def reset_parameters(self):
        
        layers = [self.sub2rel_basis.data, self.sub2rel_att,
                    self.obj2rel_basis.data, self.obj2rel_att,
                    self.rel2sub_basis.data, self.rel2sub_att,
                    self.rel2obj_basis.data, self.rel2obj_att,
                    self.entity2rel_attn.weight.data
                    ]

        for layer in layers:
            nn.init.xavier_uniform_(layer)

        if self.bias:

            biases = [self.sub2rel_bias, self.obj2rel_bias, self.rel2sub_bias, self.rel2obj_bias, self.skip_bias]

            for bias in biases:
                stdv = 1. / sqrt(bias.shape[0])
                bias.data.uniform_(-stdv, stdv)
            

    def forward(self, edge_index, nf, ef, edge_type_list, size=None):

        return self.propagate(edge_index, nf, ef, edge_type_list)



    def message(self, edgeindex_i, x_i, x_j, x_ij, edge_type, message_type):
        
        edge_mask = torch.zeros((edge_type.size(0), self.num_relations)).cuda()
        for i in range(self.num_relations):
            edge_mask[:, i] += (edge_type == i).float().cuda()
            
        if message_type == 'sub2rel':

            W = torch.matmul(self.sub2rel_att, self.sub2rel_basis.view(self.num_bases, -1))
            W = W.view(self.num_relations, self.node_in_channels, self.edge_out_channels)

            src_feat = torch.mul(x_j.unsqueeze(2), edge_mask.unsqueeze(1))
            message = src_feat

            Message = []
            for rel in range(self.num_relations):
                W_r = W[rel, ...]
                m_r = torch.matmul(message[..., rel], W_r)
                Message.append(m_r)

        elif message_type == 'obj2rel':
            W = torch.matmul(self.obj2rel_att, self.obj2rel_basis.view(self.num_bases, -1))
            W = W.view(self.num_relations, self.node_in_channels, self.edge_out_channels)

            src_feat = torch.mul(x_j.unsqueeze(2), edge_mask.unsqueeze(1))
            message = src_feat

            Message = []
            for rel in range(self.num_relations):
                W_r = W[rel, ...]
                m_r = torch.matmul(message[..., rel], W_r)

                Message.append(m_r)  
            

        elif message_type == 'rel2sub':
            W = torch.matmul(self.rel2sub_att, self.rel2sub_basis.view(self.num_bases, -1))
            W = W.view(self.num_relations, self.edge_in_channels, self.node_out_channels)

            src_feat = torch.mul(x_ij.unsqueeze(2), edge_mask.unsqueeze(1))
            message = src_feat
            Message = []
            for rel in range(self.num_relations):
                W_r = W[rel, ...]
                m_r = torch.matmul(message[..., rel], W_r)
                eij_r = F.leaky_relu(self.rel2sub_attn[rel](m_r))
                eij_r += (edge_mask[:, rel, None]-1) * 1e8
                alpha_ij_r = self.softmax(eij_r, edgeindex_i)
                alpha_ij_r *= edge_mask[:, rel, None]
                Message.append(alpha_ij_r*m_r)
        elif message_type == 'rel2obj':
            W = torch.matmul(self.rel2obj_att, self.rel2obj_basis.view(self.num_bases, -1))
            W = W.view(self.num_relations, self.edge_in_channels, self.node_out_channels)

            src_feat = torch.mul(x_ij.unsqueeze(2), edge_mask.unsqueeze(1))
            message = src_feat
            Message = []
            for rel in range(self.num_relations):
                W_r = W[rel, ...]
                m_r = torch.matmul(message[..., rel], W_r)
                eij_r = F.leaky_relu(self.rel2obj_attn[rel](m_r))
                eij_r += (edge_mask[:, rel, None]-1) * 1e8
                alpha_ij_r = self.softmax(eij_r, edgeindex_i)
                alpha_ij_r *= edge_mask[:, rel, None]
                Message.append(alpha_ij_r*m_r)

        return torch.stack(Message, -1)



    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr = None,
                  dim_size = None) -> Tensor:
        if ptr is not None:
            ptr = self.expand_left(ptr, dim=self.node_dim, dims=inputs.dim())
            return segment_csr(inputs, ptr, reduce=self.aggr)
        else:
            return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce = 'sum')

    def expand_left(self, src: torch.Tensor, dim: int, dims: int) -> torch.Tensor:
        for _ in range(dims + dim if dim < 0 else dim):
            src = src.unsqueeze(0)
        return src


    def propagate(self, edge_index, nf, ef, edge_type_list, size=None):
        size = self.__check_input__(edge_index, size)
   
        # Define dst(i) and src (j)
        x_i = nf[edge_index[1, :]]; x_j=nf[edge_index[0, :]]; x_ij=ef

        sub2rel_x_j = nf[edge_index[0, :]]
        obj2rel_x_j = nf[edge_index[1, :]]

        edge_type_rel, edge_type_rel_inv = edge_type_list

        # Generate Relation Embedding
        sub2rel_msg = torch.sum(self.message(edge_index[1, :], None, sub2rel_x_j, x_ij, edge_type_rel, 'sub2rel'), -1)
        obj2rel_msg = torch.sum(self.message(edge_index[0, :], None, obj2rel_x_j, x_ij, edge_type_rel_inv, 'obj2rel') , -1)

        # Relation Proposal Update
        sub2rel_score = self.entity2rel_attn(sub2rel_msg)
        obj2rel_score = self.entity2rel_attn(obj2rel_msg)
        entity_cat_msg = F.leaky_relu(torch.cat([sub2rel_score, obj2rel_score], 1))
        entity_score = torch.softmax(entity_cat_msg, 1)
        msg = sub2rel_msg * entity_score[:,0].view(-1,1) + obj2rel_msg * entity_score[:,1].view(-1,1)
        rel_embedding = x_ij + msg
        
        sub_msg = self.message(edge_index[0, :], x_i, None, rel_embedding, edge_type_rel_inv, 'rel2sub')
        obj_msg = self.message(edge_index[1, :], x_j, None, rel_embedding, edge_type_rel, 'rel2obj')

        # Aggregate
        node_mask_sub = torch.zeros((nf.size(0), self.num_relations)).cuda()
        node_mask_obj = torch.zeros((nf.size(0), self.num_relations)).cuda()
        sub_agg = []
        obj_agg = []

        for rel in range(self.num_relations):
            sub_agg.append(self.aggregate(sub_msg[..., rel], index = edge_index[0, :], ptr= None, dim_size=nf.size(0)))
            obj_agg.append(self.aggregate(obj_msg[..., rel], index = edge_index[1, :], ptr= None, dim_size=nf.size(0)))
            node_mask_sub[:, rel] += self.aggregate((edge_type_rel_inv == rel).float()[..., None], index=edge_index[0, :], ptr=None, dim_size=nf.size(0)).squeeze()
            node_mask_obj[:, rel] += self.aggregate((edge_type_rel == rel).float()[..., None], index=edge_index[1, :], ptr=None, dim_size=nf.size(0)).squeeze()
       
        node_mask_sub_gt = node_mask_sub.gt(0).float()
        node_mask_obj_gt = node_mask_obj.gt(0).float()

        sub_agg = torch.stack(sub_agg, -1)
        obj_agg = torch.stack(obj_agg, -1)
        

        node_mask_sub_sum = node_mask_sub_gt.sum(1).view(-1,1)
        node_mask_sub_sum[node_mask_sub_sum == 0.0] = 1.0
        node_mask_obj_sum = node_mask_obj_gt.sum(1).view(-1,1)
        node_mask_obj_sum[node_mask_obj_sum == 0.0] = 1.0
        sub_agg = torch.sum(sub_agg, -1) / node_mask_sub_sum
        obj_agg = torch.sum(obj_agg, -1) / node_mask_obj_sum

        # Update
        node_embedding = nf + (sub_agg + obj_agg)/2

        return node_embedding, rel_embedding


    def maybe_num_nodes(self, edge_index, num_nodes=None):
        if num_nodes is not None:
            return num_nodes
        elif isinstance(edge_index, Tensor):
            return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
        else:
            return max(edge_index.size(0), edge_index.size(1))


    def softmax(self, src: Tensor, index= None,
            ptr = None, num_nodes = None,
            dim: int = 0) -> Tensor:
   
        if ptr is not None:
            dim = dim + src.dim() if dim < 0 else dim
            size = ([1] * dim) + [-1]
            ptr = ptr.view(size)
            src_max = gather_csr(segment_csr(src, ptr, reduce='max'), ptr)
            out = (src - src_max).exp()
            out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)
        elif index is not None:
            N = self.maybe_num_nodes(index, num_nodes)
            src_max = scatter(src, index, dim, dim_size=N, reduce='max')
            src_max = src_max.index_select(dim, index)
            out = (src - src_max).exp()
            out_sum = scatter(out, index, dim, dim_size=N, reduce='sum')
            out_sum = out_sum.index_select(dim, index)
        else:
            raise NotImplementedError

        return out / (out_sum + 1e-16)

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)
