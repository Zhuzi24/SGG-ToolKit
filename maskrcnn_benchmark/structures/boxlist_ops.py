# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import scipy.linalg
# from mmrotate.core import  poly2obb_np
from .bounding_box import BoxList
import cv2
import numpy as np
from maskrcnn_benchmark.layers import nms as _box_nms

import cv2
import numpy as np



import cv2
import numpy as np

import cv2
import numpy as np

import torch
import cv2
import numpy as np

# def batch_poly2obb_torch_le90_8(batch_poly):
#     """Convert multiple polygons to oriented bounding boxes.

#     Args:
#         batch_poly (torch.Tensor): Shape (batch_size, 16) for each set of polygons.

#     Returns:
#         batch_obbs (torch.Tensor): Shape (batch_size, 5) for each oriented bounding box [x_ctr, y_ctr, w, h, angle].
#     """
#     num_polygons = batch_poly.size(0)
#     bboxps = batch_poly.view(num_polygons, 8, 2)

#     # Convert to NumPy array for cv2.minAreaRect
#     bboxps_np = bboxps.detach().cpu().numpy()

#     # Apply cv2.minAreaRect to the entire batch
#     rbboxes = np.array([cv2.minAreaRect(bbox.reshape(-1, 2)) for bbox in bboxps_np])

#     # Convert back to PyTorch tensor
#     # rbboxes = torch.tensor(rbboxes).to(batch_poly.device)

#     # Extract individual components (x, y, w, h, a) from each rbbox
#     x = rbboxes[:, 0, 0]
#     y = rbboxes[:, 0, 1]
#     w = rbboxes[:, 1, 0]
#     h = rbboxes[:, 1, 1]
#     a = rbboxes[:, 2] / 180 * np.pi  # Convert angle to radians

#     # Adjust w, h, and a based on conditions
#     w, h, a = adjust_dimensions(w, h, a)

#     # Stack the results into a single tensor
#     batch_obbs = torch.stack((x, y, w, h, a), dim=1)
    
#     return batch_obbs

# def adjust_dimensions(w, h, a):
#     """Adjust dimensions based on conditions."""
#     # Swap w and h if necessary
#     swap_indices = w < h
#     w[swap_indices], h[swap_indices] = h[swap_indices], w[swap_indices]
#     a[swap_indices] += np.pi / 2

#     # Ensure angle is within the range [-pi/2, pi/2]
#     a = np.where(a >= np.pi / 2, a - np.pi, a)
#     a = np.where(a < -np.pi / 2, a + np.pi, a)

#     return w, h, a

def squeeze_tensor(tensor):
    tensor = torch.squeeze(tensor)
    try:
        len(tensor)
    except TypeError:
        tensor.unsqueeze_(0)
    return tensor



def norm_angle(angle, angle_range):
    """Limit the range of angles.

    Args:
        angle (ndarray): shape(n, ).
        angle_range (Str): angle representations.

    Returns:
        angle (ndarray): shape(n, ).
    """
    if angle_range == 'oc':
        return angle
    elif angle_range == 'le135':
        return (angle + np.pi / 4) % np.pi - np.pi / 4
    elif angle_range == 'le90':
        return (angle + np.pi / 2) % np.pi - np.pi / 2
    else:
        print('Not yet implemented.')

def obb2poly_le90(rboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    N = rboxes.shape[0]
    if N == 0:
        return rboxes.new_zeros((rboxes.size(0), 8))
    x_ctr, y_ctr, width, height, angle = rboxes.select(1, 0), rboxes.select(
        1, 1), rboxes.select(1, 2), rboxes.select(1, 3), rboxes.select(1, 4)
    tl_x, tl_y, br_x, br_y = \
        -width * 0.5, -height * 0.5, \
        width * 0.5, height * 0.5
    rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y],
                        dim=0).reshape(2, 4, N).permute(2, 0, 1)
    sin, cos = torch.sin(angle), torch.cos(angle)
    M = torch.stack([cos, -sin, sin, cos], dim=0).reshape(2, 2,
                                                          N).permute(2, 0, 1)
    polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
    polys[:, ::2] += x_ctr.unsqueeze(1)
    polys[:, 1::2] += y_ctr.unsqueeze(1)
    return polys.contiguous()

# def obb2poly_le90(rboxes):  change for 0223 0244
#     """Convert oriented bounding boxes to polygons.

#     Args:
#         obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]

#     Returns:
#         polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
#     """
#     if len(rboxes.tolist()) ==  5:
#         rboxes = torch.Tensor([rboxes.tolist()])
#     N = rboxes.shape[0]
#     if N == 0:
#         return rboxes.new_zeros((rboxes.size(0), 8))
#     x_ctr, y_ctr, width, height, angle = rboxes.select(1, 0), rboxes.select(
#         1, 1), rboxes.select(1, 2), rboxes.select(1, 3), rboxes.select(1, 4)
#     tl_x, tl_y, br_x, br_y = \
#         -width * 0.5, -height * 0.5, \
#         width * 0.5, height * 0.5
#     rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y],
#                         dim=0).reshape(2, 4, N).permute(2, 0, 1)
#     sin, cos = torch.sin(angle), torch.cos(angle)
#     M = torch.stack([cos, -sin, sin, cos], dim=0).reshape(2, 2,
#                                                           N).permute(2, 0, 1)
#     polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
#     polys[:, ::2] += x_ctr.unsqueeze(1)
#     polys[:, 1::2] += y_ctr.unsqueeze(1)
#     return polys.contiguous()

def poly2obb_le90(polys):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    polys = torch.reshape(polys, [-1, 8])
    pt1, pt2, pt3, pt4 = polys[..., :8].chunk(4, 1)
    edge1 = torch.sqrt(
        torch.pow(pt1[..., 0] - pt2[..., 0], 2) +
        torch.pow(pt1[..., 1] - pt2[..., 1], 2))
    edge2 = torch.sqrt(
        torch.pow(pt2[..., 0] - pt3[..., 0], 2) +
        torch.pow(pt2[..., 1] - pt3[..., 1], 2))
    angles1 = torch.atan2((pt2[..., 1] - pt1[..., 1]),
                          (pt2[..., 0] - pt1[..., 0]))
    angles2 = torch.atan2((pt4[..., 1] - pt1[..., 1]),
                          (pt4[..., 0] - pt1[..., 0]))
    angles = polys.new_zeros(polys.shape[0])
    angles[edge1 > edge2] = angles1[edge1 > edge2]
    angles[edge1 <= edge2] = angles2[edge1 <= edge2]
    angles = norm_angle(angles, 'le90')
    x_ctr = (pt1[..., 0] + pt3[..., 0]) / 2.0
    y_ctr = (pt1[..., 1] + pt3[..., 1]) / 2.0
    edges = torch.stack([edge1, edge2], dim=1)
    width, _ = torch.max(edges, 1)
    height, _ = torch.min(edges, 1)
    return torch.stack([x_ctr, y_ctr, width, height, angles], 1)

def poly2obb_np_le90_8(poly):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    bboxps = np.array(poly).reshape((8, 2))
    rbbox = cv2.minAreaRect(bboxps)
    x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[
        2]
    # if w < 2 or h < 2:
    #     return
    a = a / 180 * np.pi
    if w < h:
        w, h = h, w
        a += np.pi / 2
    while not np.pi / 2 > a >= -np.pi / 2:
        if a >= np.pi / 2:
            a -= np.pi
        else:
            a += np.pi
    assert np.pi / 2 > a >= -np.pi / 2
    return x, y, w, h, a

def poly2obb_np_le90_8_batch_vectorized(polys_batch):
    """Convert batches of polygons to oriented bounding boxes using vectorized operations.

    Args:
        polys_batch (ndarray): Array of shape (num_samples, 8), where each row is [x0, y0, x1, y1, x2, y2, x3, y3].

    Returns:
        obbs_batch (ndarray): Array of shape (num_samples, 5), where each row is [x_ctr, y_ctr, w, h, angle].
    """
    bboxps = polys_batch.reshape((-1, 8, 2)).cpu().numpy()  # Reshape to (num_samples, 4, 2) for cv2.minAreaRect # torch.Size([10000, 8, 2]) 16 points
    rbboxes = np.array([cv2.minAreaRect(bbox) for bbox in bboxps],dtype=object)

    x, y, w, h, a =  np.array([sublist[0][0] for sublist in rbboxes]), np.array([sublist[0][1] for sublist in rbboxes]),\
                     np.array([sublist[1][0] for sublist in rbboxes]), np.array([sublist[1][1] for sublist in rbboxes]),\
                     np.array([sublist[2] for sublist in rbboxes])

    a = a / 180 * np.pi
    mask = w < h
    w[mask], h[mask] = h[mask], w[mask]
    ## check
    mask_wh =  w < h
    assert sum(mask_wh) == 0

    a[mask] += np.pi / 2

    # while not np.all((np.pi / 2 > a) & (a >= -np.pi / 2)):
    # 遍历 
    mask_1 = a >= np.pi / 2
    a[mask_1] -= np.pi
    mask_2 = a < -np.pi / 2
    a[mask_2] += np.pi

    # assert np.all((np.pi / 2 > a) & (a >= -np.pi / 2))
    check1 = a >= np.pi / 2
    check2 = a < -np.pi / 2
    assert sum(check1) == 0 and sum(check2) == 0
    obbs_batch = torch.tensor(np.column_stack((x, y, w, h, a)))
    return obbs_batch

import numpy as np

def inverse_transform_rbbox(x, y, w, h, a):
    if w < h:
        w, h = h, w
        a -= np.pi / 2
    
    while not np.pi / 2 > a >= -np.pi / 2:
        if a >= np.pi / 2:
            a -= np.pi
        else:
            a += np.pi
    
    assert np.pi / 2 > a >= -np.pi / 2
    
    a = a * 180 / np.pi  # Convert angle back to degrees
    
    # rbbox = np.array([x, y, w, h, a])
    return np.array([x, y, w, h, a])




def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    keep = _box_nms(boxes, score, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode), keep


def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    # TODO maybe add an API for querying the ws / hs
    xywh_boxes = boxlist.convert("xywh").bbox
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = (
        (ws >= min_size) & (hs >= min_size)
    ).nonzero().squeeze(1)
    return boxlist[keep]

def calculate_area(boxes):
    """Calculate the area of a list of boxes.

    Arguments:
      boxes: (Tensor) bounding boxes, sized [N,4].

    Returns:
      (Tensor) The area of each box, sized [N,].
    """
    # The boxes are assumed to be in the format (xmin, ymin, xmax, ymax).
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]
    return width * height


def boxlist_iou_pat(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [N,4].

    Returns:
      (tensor) iou, sized [N,].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    # if boxlist1.size != boxlist2.size:
    #     raise RuntimeError(
    #             "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))
    # boxlist1 = boxlist1.convert("xyxy")
    # boxlist2 = boxlist2.convert("xyxy")
    N = len(boxlist1)

    area1 = calculate_area(boxlist1)
    area2 = calculate_area(boxlist2)

    box1, box2 = boxlist1, boxlist2

    lt = torch.max(box1[:, :2], box2[:, :2])  # [N,2]
    rb = torch.min(box1[:, 2:], box2[:, 2:])  # [N,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N,]

    iou = inter / (area1 + area2 - inter)
    return iou


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))
    boxlist1 = boxlist1.convert("xyxy")
    boxlist2 = boxlist2.convert("xyxy")
    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


def boxlist_union(boxlist1, boxlist2,flag1 = None,boxlist_mid = None,flag2 = False):
    
    """
    Compute the union region of two set of boxes

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [N,4].

    Returns:
      (tensor) union, sized [N,4].
    """
    if flag2 :
            assert len(boxlist1) == len(boxlist2) and boxlist1.size == boxlist2.size

            ##### 返回初始的cv2.minAreaRect的角度
            b1 = boxlist1.bbox
            b2 = boxlist2.bbox
            b1_poly = obb2poly_le90(b1)
            b2_poly = obb2poly_le90(b2)
            b_16 = torch.cat((b1_poly, b2_poly), dim=1).cpu()
            # b_16 = torch.cat((b1_poly, b2_poly), dim=1)
            # uniom = batch_poly2obb_torch_le90_8(b_16)
            out = poly2obb_np_le90_8_batch_vectorized(b_16).float()
            
            # uni = []
            # for k in range(len(b_16)):
            # #    if k == 52:
            # #        t = 1
            #    u1 = poly2obb_np_le90_8(b_16[k])
            #    uni.append(torch.tensor(u1))
            # uniom = torch.stack(uni)
            ##  check 
            # sum(uniom ==  out)
            # tensor([10000, 10000, 10000, 10000, 10000])
            # sum(sum(uniom ==  out))
            # tensor(50000)

    
            return BoxList(out, boxlist1.size, "xywha")

    
    else:
        # if flag1  is not None:
        #     assert len(boxlist1) == len(boxlist2) ==  len(boxlist_mid) and boxlist1.size == boxlist2.size == boxlist_mid.size
        #     boxlist1 = boxlist1.convert("xyxy")
        #     boxlist2 = boxlist2.convert("xyxy")
        #     boxlist_mid = boxlist_mid.convert("xyxy")
        #     union_box = torch.cat((
        #         torch.min(torch.min(boxlist1.bbox[:,:2], boxlist2.bbox[:,:2]),boxlist_mid.bbox[:,:2]),
        #         torch.max(torch.max(boxlist1.bbox[:,2:], boxlist2.bbox[:,2:]),boxlist_mid.bbox[:,:2]),
        #         ),dim=1)
        #     return BoxList(union_box, boxlist1.size, "xyxy")
        
        # else:
            assert len(boxlist1) == len(boxlist2) and boxlist1.size == boxlist2.size
            boxlist1 = boxlist1.convert("xyxy")
            boxlist2 = boxlist2.convert("xyxy")
            union_box = torch.cat((
                torch.min(boxlist1.bbox[:,:2], boxlist2.bbox[:,:2]),
                torch.max(boxlist1.bbox[:,2:], boxlist2.bbox[:,2:])
                ),dim=1)
            return BoxList(union_box, boxlist1.size, "xyxy")

def boxlist_intersection(boxlist1, boxlist2):
    """
    Compute the intersection region of two set of boxes

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [N,4].

    Returns:
      (tensor) intersection, sized [N,4].
    """
    assert len(boxlist1) == len(boxlist2) and boxlist1.size == boxlist2.size
    boxlist1 = boxlist1.convert("xyxy")
    boxlist2 = boxlist2.convert("xyxy")
    inter_box = torch.cat((
        torch.max(boxlist1.bbox[:,:2], boxlist2.bbox[:,:2]),
        torch.min(boxlist1.bbox[:,2:], boxlist2.bbox[:,2:])
        ),dim=1)
    invalid_bbox = torch.max((inter_box[:,0] >= inter_box[:,2]).long(), (inter_box[:,1] >= inter_box[:,3]).long())
    inter_box[invalid_bbox > 0] = 0
    return BoxList(inter_box, boxlist1.size, "xyxy")

# TODO redundant, remove
def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

    for field in fields:
        if field in bboxes[0].triplet_extra_fields:
            triplet_list = [bbox.get_field(field).numpy() for bbox in bboxes]
            data = torch.from_numpy(scipy.linalg.block_diag(*triplet_list))
            cat_boxes.add_field(field, data, is_triplet=True)
        else:
            data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
            cat_boxes.add_field(field, data)

    return cat_boxes
