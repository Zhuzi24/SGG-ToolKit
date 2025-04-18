# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops.nms import batched_nms
import copy
from mmdet.core.bbox.iou_calculators import bbox_overlaps
new1 =  [48, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

from itertools import chain

def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   return_inds=False,
                   large = False):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_cfg (dict): a dict that contains the arguments of nms operations
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.

    Returns:
        tuple: (dets, labels, indices (optional)), tensors of shape (k, 5),
            (k), and (k). Dets are boxes with scores. Labels are 0-based.
    """
    num_classes = multi_scores.size(1) - 1 ## 26
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
        t3_bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)

    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)
        # t3_bboxes = multi_bboxes[:, None].expand(
        # multi_scores.size(0), num_classes, 4)

    scores = multi_scores[:, :-1]

    t3_scores = multi_scores

    labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
    labels = labels.view(1, -1).expand_as(scores)

    t3_labels = torch.arange(num_classes+1, dtype=torch.long)
    t3_labels = t3_labels.view(1, -1).expand_as(t3_scores)

    bboxes = bboxes.reshape(-1, 4) ## 1000,26,4
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    t3_bboxes  = t3_bboxes.reshape(-1, 4)  ## 1000*26,4
    t3_bboxes_tem_all = copy.deepcopy(t3_bboxes )
    t3_labels = t3_labels.reshape(-1)

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        # remove low scoring boxes
        valid_mask = scores > score_thr
    # multiply score_factor after threshold to preserve more bboxes, improve
    # mAP by 1% for YOLOv3
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.view(-1, 1).expand(
            multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
        if large:
            box_id = [[k]*48 for k in range(int(t3_bboxes_tem_all.shape[0]/48))]
            box_id = list(chain(*box_id))
            sel_box_id = [ int(box_id[inds_tem])  for inds_tem in inds]

    else:
        # TensorRT NMS plugin has invalid output filled with -1
        # add dummy data to make detection output correct.
        bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 4)], dim=0)
        scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
        labels = torch.cat([labels, labels.new_zeros(1)], dim=0)

    if bboxes.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        dets = torch.cat([bboxes, scores[:, None]], -1)
        if return_inds:
            return dets, labels, inds
        else:
            return dets, labels,[],[]  ### change add []

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)


    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    if large:

        # new_bboxes_for_nms = bboxes_for_nms[keep]
        select = [sel_box_id[kk] for kk in keep]  ## 涉及到的框
        # print(t3_bboxes_tem_all.shape)
        bboxes_for_nms_all = t3_bboxes_tem_all.reshape((int(t3_bboxes_tem_all.shape[0]/48),48,4))
        tensor_27 = torch.zeros((int(t3_bboxes_tem_all.shape[0]/48), 49, 4))
        tensor_27[:, 1:, :] = bboxes_for_nms_all
        tensor_27[:, 0, :] =bboxes_for_nms_all[:, 0, :]


        select_bboxes_for_nms_all =  tensor_27[select,:,:]
      
        # select_bboxes_for_nms_all =  select_bboxes_for_nms_all[:,new1,:]


    if return_inds:
        return dets, labels[keep], inds[keep]
    else:
        if large:
            return dets, labels[keep], select, select_bboxes_for_nms_all
        else:
            return dets, labels[keep]


def fast_nms(multi_bboxes,
             multi_scores,
             multi_coeffs,
             score_thr,
             iou_thr,
             top_k,
             max_num=-1):
    """Fast NMS in `YOLACT <https://arxiv.org/abs/1904.02689>`_.

    Fast NMS allows already-removed detections to suppress other detections so
    that every instance can be decided to be kept or discarded in parallel,
    which is not possible in traditional NMS. This relaxation allows us to
    implement Fast NMS entirely in standard GPU-accelerated matrix operations.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class+1), where the last column
            contains scores of the background class, but this will be ignored.
        multi_coeffs (Tensor): shape (n, #class*coeffs_dim).
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        iou_thr (float): IoU threshold to be considered as conflicted.
        top_k (int): if there are more than top_k bboxes before NMS,
            only top top_k will be kept.
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept. If -1, keep all the bboxes.
            Default: -1.

    Returns:
        tuple: (dets, labels, coefficients), tensors of shape (k, 5), (k, 1),
            and (k, coeffs_dim). Dets are boxes with scores.
            Labels are 0-based.
    """

    scores = multi_scores[:, :-1].t()  # [#class, n]
    scores, idx = scores.sort(1, descending=True)

    idx = idx[:, :top_k].contiguous()
    scores = scores[:, :top_k]  # [#class, topk]
    num_classes, num_dets = idx.size()
    boxes = multi_bboxes[idx.view(-1), :].view(num_classes, num_dets, 4)
    coeffs = multi_coeffs[idx.view(-1), :].view(num_classes, num_dets, -1)

    iou = bbox_overlaps(boxes, boxes)  # [#class, topk, topk]
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=1)

    # Now just filter out the ones higher than the threshold
    keep = iou_max <= iou_thr

    # Second thresholding introduces 0.2 mAP gain at negligible time cost
    keep *= scores > score_thr

    # Assign each kept detection to its corresponding class
    classes = torch.arange(
        num_classes, device=boxes.device)[:, None].expand_as(keep)
    classes = classes[keep]

    boxes = boxes[keep]
    coeffs = coeffs[keep]
    scores = scores[keep]

    # Only keep the top max_num highest scores across all classes
    scores, idx = scores.sort(0, descending=True)
    if max_num > 0:
        idx = idx[:max_num]
        scores = scores[:max_num]

    classes = classes[idx]
    boxes = boxes[idx]
    coeffs = coeffs[idx]

    cls_dets = torch.cat([boxes, scores[:, None]], dim=1)
    return cls_dets, classes, coeffs
