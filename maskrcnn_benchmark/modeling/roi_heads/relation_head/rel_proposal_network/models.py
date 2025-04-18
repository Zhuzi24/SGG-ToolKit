from maskrcnn_benchmark.modeling.roi_heads.relation_head.rel_proposal_network.loss import FocalLoss
import math

import ipdb
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.config import cfg
# from maskrcnn_benchmark.data import get_dataset_statistics
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_motifs import (
    obj_edge_vectors,
    encode_box_info,
    encode_orientedbox_info,
)
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou, squeeze_tensor
from maskrcnn_benchmark.utils.global_buffer import store_data


def gt_rel_proposal_matching(proposals, targets, fg_thres, require_overlap):
    """

    :param proposals:
    :param targets:
    :param fg_thres:
    :param require_overlap:
    :return:
        fg_pair_matrixs the box pairs that both box are matching with gt ground-truth
        prop_relatedness_matrixs: the box pairs that both boxes are matching with ground-truth relationship
    """
    assert targets is not None
    prop_relatedness_matrixs = []
    fg_pair_matrixs = []
    for img_id, (proposal, target) in enumerate(zip(proposals, targets)):
        device = proposal.bbox.device
        tgt_rel_matrix = target.get_field("relation")

        ious = boxlist_iou(target, proposal)
        is_match = ious > fg_thres
        locating_match = (ious > fg_thres).nonzero(as_tuple=False) 
        locating_match_stat = torch.zeros((len(proposal)), device=device)
        locating_match_stat[locating_match[:, 1]] = 1
        proposal.add_field("locating_match", locating_match_stat)

        # Proposal self IoU to filter non-overlap
        prp_self_iou = boxlist_iou(proposal, proposal)  
        if require_overlap:
            fg_boxpair_mat = (
                (prp_self_iou > 0) & (prp_self_iou < 1)
            ).long()  # not self & intersect
        else:
            num_prp = len(proposal)
            # [prp, prp] mark the affinity relation between the det prediction
            fg_boxpair_mat = (
                torch.ones((num_prp, num_prp), device=device).long()
                - torch.eye(num_prp, device=device).long()
            )
        # only select relations between fg proposals
        fg_boxpair_mat[locating_match_stat == 0] = 0
        fg_boxpair_mat[:, locating_match_stat == 0] = 0

        fg_pair_matrixs.append(fg_boxpair_mat)

        tgt_pair_idxs = torch.nonzero(tgt_rel_matrix != 0)

        assert tgt_pair_idxs.shape[1] == 2
        tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
        tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
        tgt_rel_labs = tgt_rel_matrix[tgt_head_idxs, tgt_tail_idxs].contiguous().view(-1)

        num_tgt_rels = tgt_rel_labs.shape[0]
        # generate binary prp mask
        num_prp = len(proposal)
        # num_tgt_rel, num_prp (matched prp head)
        binary_prp_head = is_match[tgt_head_idxs]
        # num_tgt_rel, num_prp (matched prp tail)
        binary_prp_tail = is_match[tgt_tail_idxs]
        # mark the box pair who overlaps with the gt relation box pairs
        binary_rel_mat = torch.zeros((num_prp, num_prp), device=device).long()

        fg_rel_triplets = []
        for i in range(num_tgt_rels):
            # generate binary prp mask
            bi_match_head = torch.nonzero(binary_prp_head[i] > 0)
            bi_match_tail = torch.nonzero(binary_prp_tail[i] > 0)

            num_bi_head = bi_match_head.shape[0]
            num_bi_tail = bi_match_tail.shape[0]
            if num_bi_head > 0 and num_bi_tail > 0:
                bi_match_head = (
                    bi_match_head.view(1, num_bi_head)
                    .expand(num_bi_tail, num_bi_head)
                    .contiguous()
                )
                bi_match_tail = (
                    bi_match_tail.view(num_bi_tail, 1)
                    .expand(num_bi_tail, num_bi_head)
                    .contiguous()
                )
                # binary rel only consider related or not, so its symmetric
                binary_rel_mat[bi_match_head.view(-1), bi_match_tail.view(-1)] = 1
                binary_rel_mat[bi_match_tail.view(-1), bi_match_head.view(-1)] = 1

        prop_relatedness_matrixs.append(binary_rel_mat)

    return fg_pair_matrixs, prop_relatedness_matrixs


class RelationProposalModel(nn.Module):
    def __init__(self, cfg):
        super(RelationProposalModel, self).__init__()
        self.cfg = cfg
        self.num_obj_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.embed_dim = cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.geometry_feat_dim = 128
        self.roi_feat_dim = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.hidden_dim = 512

        # mult_head_att = torch.nn.MultiheadAttention
        from maskrcnn_benchmark.data import get_dataset_statistics
        statistics = get_dataset_statistics(cfg)
        obj_classes, rel_classes = statistics["obj_classes"], statistics["rel_classes"]

        obj_embed_vecs = obj_edge_vectors(
            obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim
        )
        self.obj_sem_embed = nn.Embedding(self.num_obj_classes, self.embed_dim)

        with torch.no_grad():
            self.obj_sem_embed.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.obj_pos_embed = nn.Sequential(
            nn.Linear(9, self.geometry_feat_dim),
            # nn.BatchNorm1d(self.geometry_feat_dim, momentum=0.001),
            nn.ReLU(inplace=True),
            nn.Linear(self.geometry_feat_dim, self.geometry_feat_dim),
        )

        self.proposal_relness_cls_fc = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim * 2 + self.geometry_feat_dim * 2, 512),
            nn.BatchNorm1d(512, momentum=0.001),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )

        self.visual_features_on = (
            cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.VISUAL_FEATURES_ON
        )
        self.ignore_fg_pairs = (
            cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.IGNORE_FOREGROUND_BOXES_PAIRS
        )

        if self.visual_features_on:
            self.obj_vis_embed = nn.Sequential(
                nn.Linear(self.roi_feat_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim, momentum=0.001),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )

            self.sub_vis_embed = nn.Sequential(
                nn.Linear(self.roi_feat_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim, momentum=0.001),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.subj_self_att = MultiHeadAttention(heads=8, d_model=self.hidden_dim)
            self.obj_self_att = MultiHeadAttention(heads=8, d_model=self.hidden_dim)

        self.loss_eval = FocalLoss(alpha=1, gamma=2, logits=True)
        # self.loss_eval = nn.BCEWithLogitsLoss()

    def _train_sampling(self, proposal, pair_idx, fg_box_pair_matrixs, rel_label):
        """
        sampling the fg and bg pairs for training

        :param proposal:
        :param fg_box_pair_matrixs: mark the box pairs whose location match with the gt relationship pair
        :return:
        """
        assert fg_box_pair_matrixs is not None
        supervision_mat = torch.zeros_like(fg_box_pair_matrixs, dtype = torch.float16) # Revise
        gt_pair_idx = squeeze_tensor(
            torch.nonzero(rel_label > 0)
        )  # the resampling ignored gt will be ignored by relpn too
        # gt_pair_idx = squeeze_tensor(torch.nonzero(rel_label != 0))

        supervision_mat[pair_idx[gt_pair_idx, 0], pair_idx[gt_pair_idx, 1]] = 1
        supervision_mat[pair_idx[gt_pair_idx, 1], pair_idx[gt_pair_idx, 0]] = 1

        fg_box_pair_idx = squeeze_tensor(
            torch.nonzero((fg_box_pair_matrixs - supervision_mat) == 1)
        )
        supervision_mat[fg_box_pair_idx[:, 0], fg_box_pair_idx[:, 1]] = 0.2
        supervision_mat[fg_box_pair_idx[:, 1], fg_box_pair_idx[:, 0]] = 0.2
        # totally background pairs
        bg_pair_idx = squeeze_tensor(torch.nonzero(supervision_mat == 0))
        prop_scores = proposal.get_field("pred_scores")

        fg_box_pair_num = fg_box_pair_idx.shape[0]
        perm = torch.randperm(fg_box_pair_num)[: fg_box_pair_num // 2]
        selected_fg_box_pair_idx = fg_box_pair_idx[perm].to(bg_pair_idx.device)

        proposals_quality = prop_scores[bg_pair_idx[:, 0]] * prop_scores[bg_pair_idx[:, 1]]
        _, sorted_idx = torch.sort(proposals_quality, descending=True)
        bg_pair_num = gt_pair_idx.shape[0]
        bg_pair_num = bg_pair_num if bg_pair_num > 10 else 10
        bg_pair_idx = bg_pair_idx[sorted_idx][: int(bg_pair_num * 2)]
        # random select from a larger range
        perm = torch.randperm(bg_pair_idx.shape[0])[:bg_pair_num]
        bg_pair_idx = bg_pair_idx[perm]

        if self.ignore_fg_pairs:
            selected_pair_idx = torch.cat(
                (
                    pair_idx[gt_pair_idx],
                    bg_pair_idx,
                )
            )
        else:
            selected_pair_idx = torch.cat(
                (
                    pair_idx[gt_pair_idx],
                    selected_fg_box_pair_idx,
                    bg_pair_idx,
                )
            )

        return (
            selected_pair_idx,
            supervision_mat[selected_pair_idx[:, 0], selected_pair_idx[:, 1]],
        )

    def forward(
        self,
        inst_proposals,
        inst_roi_feat,
        rel_pair_idxs,
        rel_labels=None,
        fg_boxpair_matrixs=None,
        gt_rel_boxpair_matrixs=None,
    ):
        prop_relatedness_matrixs = []
        losses = []
        inst_roi_feat = torch.split(inst_roi_feat, [len(p) for p in inst_proposals], dim=0)

        for img_id, (proposal, roi_feats, pair_idx) in enumerate(
            zip(inst_proposals, inst_roi_feat, rel_pair_idxs)
        ):
            pred_logits = proposal.get_field("predict_logits").detach()
            device = proposal.bbox.device
            pred_rel_matrix = torch.zeros(
                (len(proposal), len(proposal)), device=device, dtype=pred_logits.dtype
            )

            if self.training:
                assert fg_boxpair_matrixs is not None
                assert rel_labels is not None
                fg_boxpair_mat = fg_boxpair_matrixs[img_id]
                pair_idx, relness_labels = self._train_sampling(
                    proposal, pair_idx, fg_boxpair_mat, rel_labels[img_id]
                )

            if proposal.mode == "xywha":
                pos_embed = self.obj_pos_embed(
                    encode_orientedbox_info(
                        [
                            proposal,
                        ]
                    ))
            else:
                pos_embed = self.obj_pos_embed(
                    encode_box_info(
                        [
                            proposal,
                        ]
                    )
            )

            # pos_embed = self.obj_pos_embed(
            #     encode_box_info(
            #         [
            #             proposal,
            #         ]
            #     )
            # )
            obj_sem_embed = F.softmax(pred_logits, dim=1) @ self.obj_sem_embed.weight
            rel_prop_repre = torch.cat(
                (
                    pos_embed[pair_idx[:, 0]],
                    obj_sem_embed[pair_idx[:, 0]],
                    pos_embed[pair_idx[:, 1]],
                    obj_sem_embed[pair_idx[:, 1]],
                ),
                dim=1,
            )
            relness = squeeze_tensor(self.proposal_relness_cls_fc(rel_prop_repre))

            if self.visual_features_on:
                sub_roi_feat = self.sub_vis_embed(roi_feats)
                sub_roi_feat = self.subj_self_att(
                    sub_roi_feat, sub_roi_feat, sub_roi_feat
                ).squeeze(1)
                obj_roi_feat = self.obj_vis_embed(roi_feats)
                obj_roi_feat = self.obj_self_att(
                    obj_roi_feat, obj_roi_feat, obj_roi_feat
                ).squeeze(1)
                visual_relness_scores = torch.mm(sub_roi_feat, obj_roi_feat.t())  # k x k
                visual_relness_scores = visual_relness_scores[pair_idx[:, 0], pair_idx[:, 1]]
                relness += visual_relness_scores

            if self.training:
                # accumulate the loss with the foreground background sampling
                if len(relness.view(-1, 1)) != 0:
                    loss = self.loss_eval(
                        relness.view(-1, 1), relness_labels.view(-1, 1).float()
                    )
                    losses.append(loss)

                relness = torch.sigmoid(relness)
                pred_rel_matrix[pair_idx[:, 0], pair_idx[:, 1]] = relness

                # add the gt pair on prediction in training phrase for more stable training
                prop_relatedness_matrixs.append(
                    pred_rel_matrix * 0.8 + gt_rel_boxpair_matrixs[img_id] * 0.2
                )
            else:

                relness = torch.sigmoid(relness)
                pred_rel_matrix[pair_idx[:, 0], pair_idx[:, 1]] = relness

                prop_relatedness_matrixs.append(pred_rel_matrix)


            if cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.EVAL_MODEL_AUC:
                if self.training:
                    y = relness_labels.detach().long().cpu().numpy()
                    pred = relness.detach().cpu().numpy()

                    store_data("rel_pn-train_y", y)
                    store_data("rel_pn-train_pred", pred)

                else:
                    fg_boxpair_mat = gt_rel_boxpair_matrixs[img_id]
                    y = fg_boxpair_mat.view(-1).detach().long().cpu().numpy()
                    pred = pred_rel_matrix.view(-1).detach().cpu().numpy()

                    store_data("rel_pn-test_y", y)
                    store_data("rel_pn-test_pred", pred)

        if self.training:
            assert len(losses) > 0
            losses = torch.mean(torch.stack(losses))
        else:
            losses = None

        return prop_relatedness_matrixs, losses


def reverse_sigmoid(x):
    new_x = x.clone()
    new_x[x > 0.999] = x[x > 0.999] - (x[x > 0.999].clone().detach() - 0.999)
    new_x[x < 0.001] = x[x < 0.001] + (-x[x < 0.001].clone().detach() + 0.001)
    return torch.log((new_x) / (1 - (new_x)))


# @registry.RELATION_CONFIDENCE_AWARE_MODULES.register("PreClassifierInstFeatureRelPN")
# class PreClassifierInstFeatureRelPN(nn.Module):
#     def __init__(self, input_dim):
#         super(PreClassifierInstFeatureRelPN, self).__init__()
#         self.cfg = cfg
#         self.num_rel_cls = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

#         self.binary_predictor = False

#         self.input_dim = input_dim
#         self.num_obj_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
#         self.embed_dim = cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
#         self.geometry_feat_dim = 128
#         self.roi_feat_dim = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
#         self.hidden_dim = 512
#         from maskrcnn_benchmark.data import get_dataset_statistics
#         statistics = get_dataset_statistics(cfg)
#         obj_classes, rel_classes = statistics["obj_classes"], statistics["rel_classes"]

#         obj_embed_vecs = obj_edge_vectors(
#             obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim
#         )
#         self.obj_sem_embed = nn.Embedding(self.num_obj_classes, self.embed_dim)

#         with torch.no_grad():
#             self.obj_sem_embed.weight.copy_(obj_embed_vecs, non_blocking=True)

#         self.obj_pos_embed = nn.Sequential(
#             nn.Linear(9, self.geometry_feat_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.geometry_feat_dim, self.geometry_feat_dim),
#         )

#         if self.binary_predictor:
#             self.out_dim = 1
#         else:
#             self.out_dim = self.num_rel_cls - 1

#         self.proposal_relness_cls_fc = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.Linear(self.embed_dim * 2 + self.geometry_feat_dim * 2, 512),
#             nn.LayerNorm(512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, self.out_dim),
#         )

#         self.visual_features_on = (
#             cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.VISUAL_FEATURES_ON
#         )

#         if self.visual_features_on:
#             self.obj_vis_embed = nn.Sequential(
#                 nn.Linear(self.input_dim, self.hidden_dim),
#                 nn.BatchNorm1d(self.hidden_dim, momentum=0.001),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(self.hidden_dim, self.hidden_dim),
#             )

#             self.sub_vis_embed = nn.Sequential(
#                 nn.Linear(self.input_dim, self.hidden_dim),
#                 nn.BatchNorm1d(self.hidden_dim, momentum=0.001),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(self.hidden_dim, self.hidden_dim),
#             )
#             self.subj_self_att = MultiHeadAttention(heads=8, d_model=self.hidden_dim)
#             self.obj_self_att = MultiHeadAttention(heads=8, d_model=self.hidden_dim)

#         self.proposal_relness_cls_vis_feat_fc = nn.Sequential(
#             nn.BatchNorm1d(self.hidden_dim, momentum=0.001),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.hidden_dim, self.out_dim),
#         )

#     def forward(
#         self,
#         inst_roi_feat,
#         entities_proposals,
#         rel_pair_inds,
#     ):

#         relness_matrix = []
#         relness_logits_batch = []
#         inst_roi_feat = torch.split(inst_roi_feat.detach(), [len(p) for p in entities_proposals], dim=0)

#         for img_id, (proposal, roi_feats, pair_idx) in enumerate(
#             zip(entities_proposals, inst_roi_feat, rel_pair_inds)
#         ):
#             pred_logits = proposal.get_field("predict_logits").detach()
#             device = proposal.bbox.device
#             pred_rel_matrix = torch.zeros(
#                 (len(proposal), len(proposal)), device=device, dtype=pred_logits.dtype
#             )

#             pos_embed = self.obj_pos_embed(
#                 encode_box_info(
#                     [
#                         proposal,
#                     ]
#                 )
#             )
#             obj_sem_embed = F.softmax(pred_logits, dim=1) @ self.obj_sem_embed.weight
#             # inst_logtis_embed = self.inst_logtis_embed(pred_logits)
#             rel_prop_repre = torch.cat(
#                 (
#                     pos_embed[pair_idx[:, 0]],
#                     obj_sem_embed[pair_idx[:, 0]],
#                     pos_embed[pair_idx[:, 1]],
#                     obj_sem_embed[pair_idx[:, 1]],
#                 ),
#                 dim=1,
#             )
#             relness_logits = squeeze_tensor(self.proposal_relness_cls_fc(rel_prop_repre))

#             if self.visual_features_on:
#                 sub_roi_feat = self.sub_vis_embed(roi_feats)
#                 sub_roi_feat = self.subj_self_att(
#                     sub_roi_feat, sub_roi_feat, sub_roi_feat
#                 ).squeeze(1)
#                 obj_roi_feat = self.obj_vis_embed(roi_feats)
#                 obj_roi_feat = self.obj_self_att(
#                     obj_roi_feat, obj_roi_feat, obj_roi_feat
#                 ).squeeze(1)

#                 visual_relness_feat = torch.index_select(
#                     sub_roi_feat, 0, pair_idx[:, 0]
#                 ) * torch.index_select(obj_roi_feat, 0, pair_idx[:, 1])

#                 visual_relness_logits = self.proposal_relness_cls_vis_feat_fc(
#                     visual_relness_feat
#                 )

#                 relness_logits = relness_logits * 0.5 + visual_relness_logits * 0.5

#             relness_scores = torch.sigmoid(relness_logits)
#             if self.binary_predictor:
#                 pred_rel_matrix[pair_idx[:, 0], pair_idx[:, 1]] = relness_scores
#             else:
#                 pred_rel_matrix[pair_idx[:, 0], pair_idx[:, 1]] = relness_scores.max(dim=1)[0]

#             relness_logits_batch.append(relness_logits)
#             relness_matrix.append(pred_rel_matrix)

#         return (
#             torch.cat(relness_logits_batch),
#             relness_matrix,
#         )


# @registry.RELATION_CONFIDENCE_AWARE_MODULES.register("GRCNNRelProp")
# class GRCNNRelProp(nn.Module):
#     def __init__(
#         self,
#         input_dim,
#     ):
#         super(GRCNNRelProp, self).__init__()
#         self.cfg = cfg
#         self.num_rel_cls = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES


#         self.predictor_type = (
#             cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.REL_AWARE_PREDICTOR_TYPE
#         )

#         self.num_obj_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES

#         self.hidden_dim = 256
#         self.out_dim = 1

#         self.sub_fc = nn.Linear(self.num_obj_classes , self.hidden_dim)
#         self.obj_fc = nn.Linear(self.num_obj_classes, self.hidden_dim)

#         self.proposal_relness_cls_fc = nn.Sequential(
#             nn.ReLU(),
#             nn.Linear(self.hidden_dim * 2, self.out_dim),
#         )

#     def forward(
#         self,
#         visual_feat,
#         entities_proposals,
#         rel_pair_inds,
#     ):

#         relness_matrix = []
#         relness_logits_batch = []



#         visual_feat_split = torch.split(visual_feat, [len(p) for p in rel_pair_inds], dim=0)

#         for img_id, (proposal, vis_feats, pair_idx) in enumerate(
#             zip(entities_proposals, visual_feat_split, rel_pair_inds)
#         ):
#             pred_logits = proposal.get_field("predict_logits").detach()
#             device = proposal.bbox.device
#             pred_rel_matrix = torch.zeros(
#                 (len(proposal), len(proposal)), device=device, dtype=pred_logits.dtype
#             )

#             obj_prob = F.softmax(pred_logits, dim=1) 
#             obj_pair_prob = torch.cat(
#                 (
#                     self.sub_fc(obj_prob[pair_idx[:, 0]]),
#                     self.obj_fc(obj_prob[pair_idx[:, 1]]),
#                 ),
#                 dim=1,
#             )

#             relness_logits = self.proposal_relness_cls_fc(obj_pair_prob)

#             relness_logits = squeeze_tensor(relness_logits)

#             relness_scores = squeeze_tensor(torch.sigmoid(relness_logits))
#             pred_rel_matrix[pair_idx[:, 0], pair_idx[:, 1]] = relness_scores

#             relness_logits_batch.append(relness_logits)
#             relness_matrix.append(pred_rel_matrix)

#         return (
#             torch.cat(relness_logits_batch),
#             relness_matrix,
#         )



@registry.RELATION_CONFIDENCE_AWARE_MODULES.register("RelAwareRelFeature")
class RelAwareRelFeature(nn.Module):
    def __init__(
        self,
        input_dim,
    ):
        super(RelAwareRelFeature, self).__init__()
        self.cfg = cfg
        self.num_rel_cls = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        self.input_dim = input_dim

        self.predictor_type = (
            cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.REL_AWARE_PREDICTOR_TYPE
        )

        self.num_obj_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.embed_dim = cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.geometry_feat_dim = 128
        self.roi_feat_dim = cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM
        self.hidden_dim = 512
        from maskrcnn_benchmark.data import get_dataset_statistics
        statistics = get_dataset_statistics(cfg)
        obj_classes, rel_classes = statistics["obj_classes"], statistics["rel_classes"]

        obj_embed_vecs = obj_edge_vectors(
            obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim
        )
        self.obj_sem_embed = nn.Embedding(self.num_obj_classes, self.embed_dim) # 200

        with torch.no_grad():
            self.obj_sem_embed.weight.copy_(obj_embed_vecs, non_blocking=True)

        self.obj_pos_embed = nn.Sequential(
            nn.Linear(9, self.geometry_feat_dim), # 128
            nn.ReLU(),
            nn.Linear(self.geometry_feat_dim, self.geometry_feat_dim),
        )

        self.visual_features_on = (
            cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.VISUAL_FEATURES_ON
        )

        self.proposal_box_feat_extract = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                self.embed_dim * 2 + self.geometry_feat_dim * 2,
                self.hidden_dim,
            ),
        )

        if self.visual_features_on:
            self.vis_embed = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.input_dim, self.hidden_dim),
            )

            self.proposal_feat_fusion = nn.Sequential(
                nn.LayerNorm(self.hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            )

        self.out_dim = self.num_rel_cls - 1

        self.proposal_relness_cls_fc = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

        if self.predictor_type == "hybrid":
            self.fusion_layer = nn.Linear(self.out_dim, 1)

    def forward(
        self,
        visual_feat,
        entities_proposals,
        rel_pair_inds,
    ):

        relness_matrix = []
        relness_logits_batch  = []

        if self.visual_features_on: # True
            visual_feat = self.vis_embed(visual_feat.detach())

        visual_feat_split = torch.split(visual_feat, [len(p) for p in rel_pair_inds], dim=0)

        for img_id, (proposal, vis_feats, pair_idx) in enumerate(
            zip(entities_proposals, visual_feat_split, rel_pair_inds)
        ):
            pred_logits = proposal.get_field("predict_logits").detach()
            device = proposal.bbox.device
            pred_rel_matrix = torch.zeros(
                (len(proposal), len(proposal)), device=device, dtype=pred_logits.dtype
            )
            if proposal.mode == "xywha":
                pos_embed = self.obj_pos_embed(
                    encode_orientedbox_info(
                        [
                            proposal,
                        ]
                    ))
            else:
                pos_embed = self.obj_pos_embed(
                    encode_box_info(
                        [
                            proposal,
                        ]
                    )
            )
            obj_sem_embed = F.softmax(pred_logits, dim=1) @ self.obj_sem_embed.weight
            rel_pair_symb_repre = torch.cat(
                (
                    pos_embed[pair_idx[:, 0]],
                    obj_sem_embed[pair_idx[:, 0]],
                    pos_embed[pair_idx[:, 1]],
                    obj_sem_embed[pair_idx[:, 1]],
                ),
                dim=1,
            )

            prop_pair_geo_feat = self.proposal_box_feat_extract(rel_pair_symb_repre)

            if self.visual_features_on: # True
                # visual_relness_feat = self.self_att(vis_feats, vis_feats, vis_feats).squeeze(1)
                visual_relness_feat = vis_feats
                rel_prop_repre = self.proposal_feat_fusion(
                    torch.cat((visual_relness_feat, prop_pair_geo_feat), dim=1)
                )
            else:
                rel_prop_repre = prop_pair_geo_feat

            relness_logits = self.proposal_relness_cls_fc(rel_prop_repre)
            relness_logits = squeeze_tensor(relness_logits)

            if self.predictor_type == "hybrid":
                relness_bin_logits = self.fusion_layer(relness_logits)

                relness_scores = squeeze_tensor(torch.sigmoid(relness_bin_logits))
                pred_rel_matrix[pair_idx[:, 0], pair_idx[:, 1]] = relness_scores

                relness_logits = torch.cat((relness_logits, relness_bin_logits), dim=1)
            elif self.predictor_type == "single":
                relness_scores = squeeze_tensor(torch.sigmoid(relness_logits))
                pred_rel_matrix[pair_idx[:, 0], pair_idx[:, 1]] = relness_scores.max(dim=1)[0]

            relness_logits_batch.append(relness_logits)

            relness_matrix.append(pred_rel_matrix)

        return (
            torch.cat(relness_logits_batch),
            relness_matrix,
        )


def make_relation_confidence_aware_module(in_channels):
    func = registry.RELATION_CONFIDENCE_AWARE_MODULES[
        cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.METHOD # RelAwareRelFeature
    ]
    return func(in_channels)


def filter_rel_pairs(relness_matrix, rel_pair_idxs, rel_labels=None):
    filtered_rel_pairs = []
    filtered_rel_labels = []
    selected_rel_pairs_idx = []
    valid_pair_num = (
        cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.PAIR_NUMS_AFTER_FILTERING
    )
    if valid_pair_num < 0:
        return rel_pair_idxs, rel_labels

    for idx, (rel_mat, rel_pair) in enumerate(zip(relness_matrix, rel_pair_idxs)):
        relness_scores = rel_mat[rel_pair[:, 0], rel_pair[:, 1]]
        _, selected_rel_prop_pairs_idx = torch.sort(relness_scores, descending=True)
        selected_rel_prop_pairs_idx = selected_rel_prop_pairs_idx[:valid_pair_num]
        selected_rel_pairs_idx.append(selected_rel_prop_pairs_idx)
        filtered_rel_pairs.append(rel_pair[selected_rel_prop_pairs_idx])
        if rel_labels is not None:
            filtered_rel_labels.append(rel_labels[idx][selected_rel_prop_pairs_idx])

    return filtered_rel_pairs, filtered_rel_labels if len(filtered_rel_labels) > 0 else None, selected_rel_pairs_idx


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        def attention(q, k, v, d_k, mask=None, dropout=None):
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
            if mask is not None:
                mask = mask.unsqueeze(1)
                scores = scores.masked_fill(mask == 0, -1e9)

            scores = F.softmax(scores, dim=-1)

            if dropout is not None:
                scores = dropout(scores)

            output = torch.matmul(scores, v)
            return output

        # calculate attention using function we will define next
        att_result = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = att_result.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)

        return output
