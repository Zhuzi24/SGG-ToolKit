# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops import nms_rotated
import copy
import numpy as np
from itertools import chain
new = [5, 1, 6, 11, 4, 10, 15, 9, 0, 3, 14, 2, 13, 8, 25, 12, 7, 19, 16, 17, 18, 20, 21, 24, 23, 22,26]
new1 = [26,0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

def multiclass_nms_rotated(multi_bboxes,
                           multi_scores,
                           score_thr,
                           nms,
                           max_num=-1,
                           score_factors=None,
                           return_inds=False,
                           large = False):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (torch.Tensor): shape (n, #class*5) or (n, 5)
        multi_scores (torch.Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms (float): Config of NMS.
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.

    Returns:
        tuple (dets, labels, indices (optional)): tensors of shape (k, 5), \
        (k), and (k). Dets are boxes with scores. Labels are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    ###

    ###
    # exclude background category
    if multi_bboxes.shape[1] > 5:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 5)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 5)
        t3_bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), multi_scores.size(1), 5)
        
    scores = multi_scores[:, :-1]
    t3_scores = multi_scores

    labels = torch.arange(num_classes, dtype=torch.long)
    labels = labels.view(1, -1).expand_as(scores)

    t3_labels = torch.arange(num_classes+1, dtype=torch.long)
    t3_labels = t3_labels.view(1, -1).expand_as(t3_scores)

    bboxes_tem = copy.deepcopy(bboxes)
    bboxes = bboxes.reshape(-1, 5) ## 2000,26,5
    t3_bboxes  = t3_bboxes.reshape(-1, 5)
    bboxes_tem_all = copy.deepcopy(bboxes)
    t3_bboxes_tem_all = copy.deepcopy(t3_bboxes )
    sc_tem = copy.deepcopy(scores)
    box_tem = copy.deepcopy(bboxes)
    

    scores = scores.reshape(-1) ## 所有合并 2000,27 to 52000
    labels = labels.reshape(-1)
    t3_labels = t3_labels.reshape(-1)
    labels_all = copy.deepcopy(labels)

    # remove low scoring boxes
    valid_mask = scores > score_thr
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.view(-1, 1).expand(
            multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
    bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds] ## 从 52000 中挑选
    
    if large:
        box_id = [[k]*26 for k in range(2000)]
        box_id = list(chain(*box_id))
        sel_box_id = [ int(box_id[inds_tem])  for inds_tem in inds]
    #### check
    # for y1 in range(len(labels)):
    # # y1,y2 in zip(labels,sel_id):
    #     if not (np.argmax(sc_tem[sel_id[y1]].cpu().numpy()) == int(labels[y1])):
    #         print(y1)s
    # ####sc_tem
        

    if bboxes.numel() == 0:
        dets = torch.cat([bboxes, scores[:, None]], -1)
        if return_inds:
            return dets, labels, inds
        else:
            return dets, labels,[],[]  ### change add []

    # Strictly, the maximum coordinates of the rotating box (x,y,w,h,a)
    # should be calculated by polygon coordinates.
    # But the conversion from rbbox to polygon will slow down the speed.
    # So we use max(x,y) + max(w,h) as max coordinate
    # which is larger than polygon max coordinate
    # max(x1, y1, x2, y2,x3, y3, x4, y4)
    
    #### 获取所有的
    # max_coordinate_all =  bboxes_tem_all[:, :2].max() + bboxes_tem_all[:, 2:4].max()  # torch.Size([52000, 2])
    # offsets_all = labels_all.to(box_tem) * (max_coordinate_all + 1) [52000]
    # bboxes_for_nms_all = bboxes_tem_all.clone() # 52000,5
    # bboxes_for_nms_all[:, :2] = bboxes_for_nms_all[:, :2] + offsets_all[:, None]

    max_coordinate_all =  t3_bboxes_tem_all[:, :2].max() + t3_bboxes_tem_all[:, 2:4].max()  # torch.Size([52000, 2])
    offsets_all = t3_labels.to(t3_bboxes_tem_all) * (max_coordinate_all + 1) # [52000]
    bboxes_for_nms_all = t3_bboxes_tem_all.clone() # 52000,5
    bboxes_for_nms_all[:, :2] = bboxes_for_nms_all[:, :2] + offsets_all[:, None]

    ####

    max_coordinate = bboxes[:, :2].max() + bboxes[:, 2:4].max()
    offsets = labels.to(bboxes) * (max_coordinate + 1)
    if bboxes.size(-1) == 5:
        bboxes_for_nms = bboxes.clone()
        bboxes_for_nms[:, :2] = bboxes_for_nms[:, :2] + offsets[:, None]
    else:
        
        bboxes_for_nms = bboxes + offsets[:, None]
    _, keep = nms_rotated(bboxes_for_nms, scores, nms.iou_thr)

    if max_num > 0:
        keep = keep[:max_num]

    bboxes = bboxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    if large:
        # new_bboxes_for_nms = bboxes_for_nms[keep]
        select = [sel_box_id[kk] for kk in keep]  ## 涉及到的框
        bboxes_for_nms_all = bboxes_for_nms_all.reshape((2000,27,5))
        select_bboxes_for_nms_all =  bboxes_for_nms_all[select,:,:]
        select_bboxes_for_nms_all =  select_bboxes_for_nms_all[:,new,:]
        select_bboxes_for_nms_all =  select_bboxes_for_nms_all[:,new1,:]

        # sc_tem[select[0]]
    if return_inds:
        return torch.cat([bboxes, scores[:, None]], 1), labels, keep
    else:
        if large:
            return torch.cat([bboxes, scores[:, None]], 1), labels, select, select_bboxes_for_nms_all
        else:
            return torch.cat([bboxes, scores[:, None]], 1), labels


def aug_multiclass_nms_rotated(merged_bboxes, merged_labels, score_thr, nms,
                               max_num, classes):
    """NMS for aug multi-class bboxes.

    Args:
        multi_bboxes (torch.Tensor): shape (n, #class*5) or (n, 5)
        multi_scores (torch.Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms (float): Config of NMS.
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        classes (int): number of classes.

    Returns:
        tuple (dets, labels): tensors of shape (k, 5), and (k). Dets are boxes
            with scores. Labels are 0-based.
    """
    bboxes, labels = [], []

    for cls in range(classes):
        cls_bboxes = merged_bboxes[merged_labels == cls]
        inds = cls_bboxes[:, -1] > score_thr
        if len(inds) == 0:
            continue
        cur_bboxes = cls_bboxes[inds, :]
        cls_dets, _ = nms_rotated(cur_bboxes[:, :5], cur_bboxes[:, -1],
                                  nms.iou_thr)
        cls_labels = merged_bboxes.new_full((cls_dets.shape[0], ),
                                            cls,
                                            dtype=torch.long)
        if cls_dets.size()[0] == 0:
            continue
        bboxes.append(cls_dets)
        labels.append(cls_labels)

    if bboxes:
        bboxes = torch.cat(bboxes)
        labels = torch.cat(labels)
        if bboxes.shape[0] > max_num:
            _, _inds = bboxes[:, -1].sort(descending=True)
            _inds = _inds[:max_num]
            bboxes = bboxes[_inds]
            labels = labels[_inds]
    else:
        bboxes = merged_bboxes.new_zeros((0, merged_bboxes.size(-1)))
        labels = merged_bboxes.new_zeros((0, ), dtype=torch.long)

    return bboxes, labels
