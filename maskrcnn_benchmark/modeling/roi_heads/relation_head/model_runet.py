from maskrcnn_benchmark.modeling import registry
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from maskrcnn_benchmark.modeling.utils import cat
from .utils_motifs import obj_edge_vectors, center_x, sort_by_score, to_onehot, get_dropout_mask, nms_overlaps, encode_box_info
from .utils_relation import layer_init
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
import scipy.stats
import random

def encode_box_info(proposals):
    """
    encode proposed box information (x1, y1, x2, y2) to
    (cx/wid, cy/hei, w/wid, h/hei, x1/wid, y1/hei, x2/wid, y2/hei, wh/wid*hei)
    """
    assert proposals[0].mode == 'xyxy'
    boxes_info = []
    for proposal in proposals:
        boxes = proposal.bbox
        img_size = proposal.size
        wid = img_size[0]
        hei = img_size[1]
        wh = boxes[:, 2:] - boxes[:, :2] + 1.0
        xy = boxes[:, :2] + 0.5 * wh
        w, h = wh.split([1,1], dim=-1)
        x, y = xy.split([1,1], dim=-1)
        x1, y1, x2, y2 = boxes.split([1,1,1,1], dim=-1)
        assert wid * hei != 0
        info = torch.cat([w/wid, h/hei, x/wid, y/hei, x1/wid, y1/hei, x2/wid, y2/hei,
                          w*h/(wid*hei)], dim=-1).view(-1, 9)
        boxes_info.append(info)

    return torch.cat(boxes_info, dim=0)

def bbox_transform_inv(boxes, gt_boxes, weights=(1.0, 1.0, 1.0, 1.0)):
    """Inverse transform that computes target bounding-box regression deltas
    given proposal boxes and ground-truth boxes. The weights argument should be
    a 4-tuple of multiplicative weights that are applied to the regression
    target.

    In older versions of this code (and in py-faster-rcnn), the weights were set
    such that the regression deltas would have unit standard deviation on the
    training dataset. Presently, rather than computing these statistics exactly,
    we use a fixed set of weights (10., 10., 5., 5.) by default. These are
    approximately the weights one would get from COCO using the previous unit
    stdev heuristic.
    """
    ex_widths = boxes[:, 2] - boxes[:, 0] + 1.0
    ex_heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ex_ctr_x = boxes[:, 0] + 0.5 * ex_widths
    ex_ctr_y = boxes[:, 1] + 0.5 * ex_heights

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

    wx, wy, ww, wh = weights
    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)

    targets = torch.stack((targets_dx, targets_dy, targets_dw,
                           targets_dh), -1)
    return targets


def get_spt_features(boxes1, boxes2, boxes_u, width, height):
    # boxes_u = boxes_union(boxes1, boxes2)
    spt_feat_1 = get_box_feature(boxes1, width, height)
    spt_feat_2 = get_box_feature(boxes2, width, height)
    spt_feat_12 = get_pair_feature(boxes1, boxes2)
    spt_feat_1u = get_pair_feature(boxes1, boxes_u)
    spt_feat_u2 = get_pair_feature(boxes_u, boxes2)
    return torch.cat((spt_feat_12, spt_feat_1u, spt_feat_u2, spt_feat_1, spt_feat_2), -1)

def get_area(boxes):
    area = (boxes[:, 2] -  boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return area


def get_pair_feature(boxes1, boxes2):
    delta_1 = bbox_transform_inv(boxes1, boxes2)
    delta_2 = bbox_transform_inv(boxes2, boxes1)
    spt_feat = torch.cat((delta_1, delta_2[:, :2]), -1)
    return spt_feat


def get_box_feature(boxes, width, height):
    f1 = boxes[:, 0] / width
    f2 = boxes[:, 1] / height
    f3 = boxes[:, 2] / width
    f4 = boxes[:, 3] / height
    f5 = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1) / (width * height)
    return torch.stack((f1, f2, f3, f4, f5), -1)


class Boxes_Encode(nn.Module):
    def __init__(self,):
        super(Boxes_Encode, self).__init__()
        self.spt_feats = nn.Sequential(
            nn.Linear(28, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.1))

    def spo_boxes(self, boxes, rel_inds):
        s_boxes = boxes[rel_inds[:, 0]]
        o_boxes = boxes[rel_inds[:, 1]]
        union_boxes = torch.cat((
            torch.min(s_boxes[:, 0:2], o_boxes[:, 0:2]),
            torch.max(s_boxes[:, 2:], o_boxes[:, 2:])
        ), 1)

        return s_boxes, o_boxes, union_boxes

    def forward(self, boxes, rel_inds, width, height):
        s_boxes, o_boxes, u_boxes = self.spo_boxes(boxes, rel_inds)
        spt_feats = get_spt_features(s_boxes, o_boxes, u_boxes, width, height)

        return self.spt_feats(spt_feats)

class Get_Atten_map_mc_clear(nn.Module):

    def __init__(self, input_dims, p):
        super(Get_Atten_map_mc_clear, self).__init__()
        self.input_dims = input_dims
        self.p = p
        self.ws = nn.Linear(self.input_dims, self.input_dims)
        self.wo = nn.Linear(self.input_dims, self.input_dims)
        self.w = nn.Linear(self.input_dims, self.p)
        # self.act = nn.ReLU(inplace=True)
        # self.act = CELU(alpha=1.3)
        # self.act = nn.Sequential()
        self.tau = 0.5
        self.tau_pm2 = 4.
        self.T = 1.


    def forward(self, obj_feats, union_feats, pair_idxs):

        norm_mat = (obj_feats[:,None,:]-obj_feats[None,:,:]).norm(dim=-1,keepdim=True) 
        n_nodes = obj_feats.shape[0]
        atten_f = self.w(self.ws(obj_feats)[pair_idxs[:, 0]] * self.wo(obj_feats)[pair_idxs[:, 1]] * union_feats)
        atten_tensor = torch.zeros(n_nodes, n_nodes, self.p).to(obj_feats)
        atten_tensor[pair_idxs[:, 0], pair_idxs[:, 1]] += atten_f
        eye_tensor = -torch.eye(n_nodes).unsqueeze(-1).repeat(1, 1, self.p).to(obj_feats) * 1e4
        atten_tensor = atten_tensor + eye_tensor
        atten_map = F.softmax(atten_tensor, dim=1)

        Omega = torch.zeros_like(atten_map)
        Omega = Omega.masked_fill_(norm_mat<self.tau, self.tau_pm2) 
        Omega = Omega.masked_fill_(torch.eye(Omega.shape[0], dtype=bool, device=Omega.device).unsqueeze(-1), 0.) 
        Omega = torch.where((norm_mat>=self.tau)&(norm_mat<self.T), norm_mat.clamp(min=1e-5).pow(-2.), Omega) 
        atten_map = Omega * atten_map
        
        return atten_map

def mc_matmul(tensor3d, mat):
    out = []
    for i in range(tensor3d.size(-1)):
        out.append(torch.mm(tensor3d[:, :, i], mat))
    return torch.cat(out, -1)

class Message_Passing4OBJ(nn.Module):

    def __init__(self, input_dims):
        super(Message_Passing4OBJ, self).__init__()
        self.input_dims = input_dims
        self.trans = nn.Sequential(nn.Linear(self.input_dims, self.input_dims*2),
                                   nn.ReLU(),
                                   nn.Linear(self.input_dims*2, self.input_dims))

        self.get_atten_tensor = Get_Atten_map_mc_clear(self.input_dims, p=1)

        self.conv = nn.Sequential(nn.Linear(self.input_dims, self.input_dims),
                                    nn.ReLU())

        self.ln1 = nn.LayerNorm(self.input_dims)
        self.ln2 = nn.LayerNorm(self.input_dims)

    def forward(self, obj_feats, phr_feats, pair_idxs):

        refined_obj_feats = []

        for iobj_feats, iphr_feats, ipair_idxs in zip(
            obj_feats, phr_feats, pair_idxs
        ):
            if not ipair_idxs.shape[0] > 1:
                refined_obj_feats.append(
                    iobj_feats
                )
                continue
            atten_tensor = self.get_atten_tensor(iobj_feats, iphr_feats, ipair_idxs)

            context_feats = torch.mm(atten_tensor.squeeze(-1), self.conv(self.ln1(iobj_feats)))

            outputs = iobj_feats + context_feats

            refined_obj_feats.append(
                F.relu(
                    outputs + self.trans(self.ln2(outputs))
                )
            )
        return refined_obj_feats

class RUNetContext(nn.Module):
    def __init__(self, config, obj_class, obj_dim=4096, embed_dim=200, hidden_dim=512):
        super(RUNetContext, self).__init__()
        self.num_obj_cls = len(obj_class)
        self.cfg = config
        # self.mode = mode
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.obj_dim = obj_dim
        self.nms_thresh = 0.7

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        self.classifier = nn.Linear(self.hidden_dim, self.num_obj_cls)

        self.obj_mps1 = Message_Passing4OBJ(self.hidden_dim)
        self.obj_mps2 = Message_Passing4OBJ(self.hidden_dim)
        self.ln = nn.LayerNorm(self.hidden_dim)

        self.merge_obj_feats = nn.Linear(self.obj_dim+self.embed_dim+128,
                                        self.hidden_dim)

        self.get_phr_feats = nn.Linear(self.obj_dim, self.hidden_dim)
        embed_vecs = obj_edge_vectors(obj_class, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)

        with torch.no_grad():
            self.obj_embed.weight.copy_(embed_vecs, non_blocking=True)

            # This probably doesn't help it much
        # self.pos_embed = nn.Sequential(*[
        #     nn.BatchNorm1d(4, momentum=0.001),
        #     nn.Linear(4, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        # ])

        self.pos_embed = nn.Sequential(*[
            nn.BatchNorm1d(5, momentum=0.001),
            nn.Linear(5, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        ])

    @staticmethod
    def center_xywh(bbox_tensor):

        return torch.cat((bbox_tensor[:, :2] + 0.5 * bbox_tensor[:, 2:],
                          bbox_tensor[:, 2:]), dim=-1)
        
        # return torch.cat((bbox_tensor[:, 0], bbox_tensor[:, 1],
        #                   bbox_tensor[:, 2], bbox_tensor[:, 1] ), dim=-1)

    def forward(self, roi_features, proposals, union_features, pair_idxs, logger=None):

        num_objs = [len(b) for b in proposals]
        num_pairs = [p.shape[0] for p in pair_idxs]

        assert len(num_objs) == len(num_pairs)

        if self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        else:
            obj_labels = None

        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_embed = self.obj_embed(obj_labels.long())
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed.weight


        # if proposals[0].mode == 'xyxy':
        #     centor_proposals = [p.convert('xywh') for p in proposals]
        # else:
        #     centor_proposals = proposals
        centor_proposals = proposals
      #  centor_proposals = [p.convert('xywh') for p in proposals] # 

        pos_embed = self.pos_embed(cat([p.bbox for p in centor_proposals], dim=0))  # self.pos_embed(cat([self.center_xywh(p.bbox) for p in centor_proposals], dim=0))

        obj_feats = self.merge_obj_feats(cat((
            roi_features, obj_embed, pos_embed
        ), dim=-1))

        phr_feats = self.get_phr_feats(union_features)

        obj_feats = obj_feats.split(num_objs, dim=0)

        phr_feats = phr_feats.split(num_pairs, dim=0)

        obj_feats = self.obj_mps1(obj_feats, phr_feats, pair_idxs)
        obj_feats = self.obj_mps2(obj_feats, phr_feats, pair_idxs)
        obj_dists = cat([self.classifier(self.ln(iobj_feats)) for iobj_feats in obj_feats], 0)

        if obj_labels is not None:
            obj_preds = obj_labels
            if not self.training or self.mode == 'predcls':
                obj_dists = to_onehot(obj_preds, self.num_obj_cls)

        elif self.mode != 'sgdet':
            obj_preds = F.softmax(obj_dists, dim=-1)[:, 1:].argmax(-1).long() + 1

        else:
            boxes_for_nms = cat([proposal.get_field('boxes_per_cls') for proposal in proposals], dim=0)
            assert len(num_objs) == 1
            is_overlap = nms_overlaps(boxes_for_nms).view(
                boxes_for_nms.size(0), boxes_for_nms.size(0), boxes_for_nms.size(1)
            ).cpu().numpy() >= self.nms_thresh

            out_dists_sampled = F.softmax(obj_dists, -1).cpu().numpy()
            out_dists_sampled[:,0] = 0

            out_commitments = obj_dists.new(obj_dists.shape[0]).fill_(0)
            for i in range(out_commitments.size(0)):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_commitments[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind,:,cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0 # This way we won't re-sample

            obj_preds = out_commitments

        return obj_dists, obj_preds, obj_feats