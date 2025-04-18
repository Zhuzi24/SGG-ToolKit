# Copyright (c) OpenMMLab. All rights reserved.
# Reference: https://github.com/jbwang1997/BboxToolkit

import itertools
from math import ceil
from multiprocessing import Manager, Pool

import torch
from mmcv.ops import nms, nms_rotated
import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

try:
    import shapely.geometry as shgeo
except ImportError:
    shgeo = None

def hbb2obb(hbboxes, version='oc'):
    """Convert horizontal bounding boxes to oriented bounding boxes.

    Args:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
        version (Str): angle representations.

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    if version == 'oc':
        results = hbb2obb_oc(hbboxes)
    elif version == 'le135':
        results = hbb2obb_le135(hbboxes)
    elif version == 'le90':
        results = hbb2obb_le90(hbboxes)
    else:
        raise NotImplementedError
    return results

def hbb2obb_oc(hbboxes):
    """Convert horizontal bounding boxes to oriented bounding boxes.

    Args:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
    y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
    w = hbboxes[..., 2] - hbboxes[..., 0]
    h = hbboxes[..., 3] - hbboxes[..., 1]
    theta = x.new_zeros(*x.shape)
    rbboxes = torch.stack([x, y, h, w, theta + np.pi / 2], dim=-1)
    return rbboxes

def hbb2obb_le135(hbboxes):
    """Convert horizontal bounding boxes to oriented bounding boxes.

    Args:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
    y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
    w = hbboxes[..., 2] - hbboxes[..., 0]
    h = hbboxes[..., 3] - hbboxes[..., 1]
    theta = x.new_zeros(*x.shape)
    obboxes1 = torch.stack([x, y, w, h, theta], dim=-1)
    obboxes2 = torch.stack([x, y, h, w, theta + np.pi / 2], dim=-1)
    obboxes = torch.where((w >= h)[..., None], obboxes1, obboxes2)
    return obboxes

def hbb2obb_le90(hbboxes):
    """Convert horizontal bounding boxes to oriented bounding boxes.

    Args:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]

    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
    y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
    w = hbboxes[..., 2] - hbboxes[..., 0]
    h = hbboxes[..., 3] - hbboxes[..., 1]
    theta = x.new_zeros(*x.shape)
    obboxes1 = torch.stack([x, y, w, h, theta], dim=-1)
    obboxes2 = torch.stack([x, y, h, w, theta - np.pi / 2], dim=-1)
    obboxes = torch.where((w >= h)[..., None], obboxes1, obboxes2)
    return obboxes


def get_sliding_window(info, sizes, gaps, img_rate_thr=0.3):
    """Get sliding windows.

    Args:
        info (dict): Dict of image's width and height.
        sizes (list): List of window's sizes.
        gaps (list): List of window's gaps.
        img_rate_thr (float): Threshold of window area divided by image area.

    Returns:
        list[np.array]: Information of valid windows.
    """
    eps = 0.01
    windows = []
    width, height = info['width'], info['height']
    for size, gap in zip(sizes, gaps):
        assert size > gap, f'invaild size gap pair [{size} {gap}]'
        step = size - gap

        x_num = 1 if width <= size else ceil((width - size) / step + 1)
        x_start = [step * i for i in range(x_num)]
        if len(x_start) > 1 and x_start[-1] + size > width:
            x_start[-1] = width - size

        y_num = 1 if height <= size else ceil((height - size) / step + 1)
        y_start = [step * i for i in range(y_num)]
        if len(y_start) > 1 and y_start[-1] + size > height:
            y_start[-1] = height - size

        start = np.array(
            list(itertools.product(x_start, y_start)), dtype=np.int32)
        stop = start + size
        windows.append(np.concatenate([start, stop], axis=1))
    windows = np.concatenate(windows, axis=0)

    img_in_wins = windows.copy()
    img_in_wins[:, 0::2] = np.clip(img_in_wins[:, 0::2], 0, width)
    img_in_wins[:, 1::2] = np.clip(img_in_wins[:, 1::2], 0, height)
    img_areas = (img_in_wins[:, 2] - img_in_wins[:, 0]) * \
                (img_in_wins[:, 3] - img_in_wins[:, 1])
    win_areas = (windows[:, 2] - windows[:, 0]) * \
                (windows[:, 3] - windows[:, 1])
    img_rates = img_areas / win_areas
    if not (img_rates > img_rate_thr).any():
        max_rate = img_rates.max()
        img_rates[abs(img_rates - max_rate) < eps] = 1
    return windows[img_rates > img_rate_thr]


def poly2hbb(polys):
    """Convert polygons to horizontal bboxes.

    Args:
        polys (np.array): Polygons with shape (N, 8)

    Returns:
        np.array: Horizontal bboxes.
    """
    shape = polys.shape
    polys = polys.reshape(*shape[:-1], shape[-1] // 2, 2)
    lt_point = np.min(polys, axis=-2)
    rb_point = np.max(polys, axis=-2)
    return np.concatenate([lt_point, rb_point], axis=-1)


def bbox_overlaps_iof(bboxes1, bboxes2, eps=1e-6):
    """Compute bbox overlaps (iof).

    Args:
        bboxes1 (np.array): Horizontal bboxes1.
        bboxes2 (np.array): Horizontal bboxes2.
        eps (float, optional): Defaults to 1e-6.

    Returns:
        np.array: Overlaps.
    """
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]

    if rows * cols == 0:
        return np.zeros((rows, cols), dtype=np.float32)

    hbboxes1 = poly2hbb(bboxes1)
    hbboxes2 = bboxes2
    hbboxes1 = hbboxes1[:, None, :]
    lt = np.maximum(hbboxes1[..., :2], hbboxes2[..., :2])
    rb = np.minimum(hbboxes1[..., 2:], hbboxes2[..., 2:])
    wh = np.clip(rb - lt, 0, np.inf)
    h_overlaps = wh[..., 0] * wh[..., 1]

    l, t, r, b = [bboxes2[..., i] for i in range(4)]
    polys2 = np.stack([l, t, r, t, r, b, l, b], axis=-1)
    if shgeo is None:
        raise ImportError('Please run "pip install shapely" '
                          'to install shapely first.')
    sg_polys1 = [shgeo.Polygon(p) for p in bboxes1.reshape(rows, -1, 2)]
    sg_polys2 = [shgeo.Polygon(p) for p in polys2.reshape(cols, -1, 2)]
    overlaps = np.zeros(h_overlaps.shape)
    for p in zip(*np.nonzero(h_overlaps)):
        overlaps[p] = sg_polys1[p[0]].intersection(sg_polys2[p[-1]]).area
    unions = np.array([p.area for p in sg_polys1], dtype=np.float32)
    unions = unions[..., None]

    unions = np.clip(unions, eps, np.inf)
    outputs = overlaps / unions
    if outputs.ndim == 1:
        outputs = outputs[..., None]
    return outputs


def get_window_obj(info, windows, iof_thr=0.1):
    """

    Args:
        info (dict): Dict of bbox annotations.
        windows (np.array): information of sliding windows.
        iof_thr (float): Threshold of overlaps between bbox and window.

    Returns:
        list[dict]: List of bbox annotations of every window.
    """
    bboxes = info['ann']['bboxes']
    # bboxes = info['ann']['bboxes']
    window_anns = []
    if bboxes is not None:
        iofs = bbox_overlaps_iof(bboxes, windows)

        for i in range(windows.shape[0]):
            win_iofs = iofs[:, i]

            pos_inds = np.nonzero(win_iofs >= iof_thr)[0].tolist()

            win_ann = dict()
            for k, v in info['ann'].items():
                try:
                    win_ann[k] = v[pos_inds]
                except TypeError:
                    win_ann[k] = [v[i] for i in pos_inds]
            win_ann['trunc'] = win_iofs[pos_inds] < 1
            window_anns.append(win_ann)

    else:
        for i in range(windows.shape[0]):
            win_ann = dict()
            win_ann['trunc']=[]
            window_anns.append(win_ann)
    return window_anns


def crop_and_save_img(info, windows, window_anns, img, no_padding,
                      padding_value):
    """

    Args:
        info (dict): Image's information.
        windows (np.array): information of sliding windows.
        window_anns (list[dict]): List of bbox annotations of every window.
        img (tensor): Full images.
        no_padding (bool): If True, no padding.
        padding_value (tuple[int|float]): Padding value.

    Returns:
        list[dict]: Information of paths.
    """
    if torch.is_tensor(img):
        img = img.clone()

    patchs=[]
    patch_infos = []
    for i in range(windows.shape[0]):
        patch_info = dict()
        for k, v in info.items():
            if k not in ['id', 'fileanme', 'width', 'height', 'ann']:
                # 
                patch_info[k] = v

        window = windows[i]
        x_start, y_start, x_stop, y_stop = window.tolist()
        patch_info['x_start'] = x_start
        patch_info['y_start'] = y_start
        # patch_info['id'] = \
        #     info['id'] + '__' + str(x_stop - x_start) + \
        #     '__' + str(x_start) + '___' + str(y_start)
        # patch_info['ori_id'] = info['id']

        ann = window_anns[i]
        ann['bboxes'] = translate(ann['bboxes'], -x_start, -y_start)
        patch_info['ann'] = ann

        # 
        patch = img[:, y_start:y_stop, x_start:x_stop]

        if not no_padding:
            height = y_stop - y_start
            width = x_stop - x_start

            if height > patch.shape[1] or width > patch.shape[2]:
            # if height > patch.shape[0] or width > patch.shape[1]:
                padding_patch = np.empty((height, width, patch.shape[-1]),
                                         dtype=np.uint8)
                if not isinstance(padding_value, (int, float)):
                    assert len(padding_value) == patch.shape[-1]
                padding_patch[...] = padding_value
                padding_patch[:patch.shape[0], :patch.shape[1], ...] = patch
                patch = padding_patch
        patch_info['height'] = patch.shape[0]
        patch_info['width'] = patch.shape[1]

        bboxes_num = patch_info['ann']['bboxes'].shape[0]
        patch_label = []

        if bboxes_num == 0:
            # patch_info['labels']=[-1]  # 
            patch_label=[-1]
            pass
        else:
            for idx in range(bboxes_num):
                obj = patch_info['ann']
                patch_label.append(patch_info['labels'][idx])

        patch_info['labels'] = patch_label
        patch_infos.append(patch_info)
        patchs.append(patch)

    return patchs,patch_infos

def crop_img_withoutann(info, windows, img, no_padding,padding_value):
    """

    Args:
        info (dict): Image's information.
        windows (np.array): information of sliding windows.
        window_anns (list[dict]): List of bbox annotations of every window.
        img (tensor): Full images.
        no_padding (bool): If True, no padding.
        padding_value (tuple[int|float]): Padding value.

    Returns:
        list[dict]: Information of paths.
    """
    img = img.clone()
    patchs = []
    patch_infos = []
    for i in range(windows.shape[0]):
        patch_info = dict()
        for k, v in info.items():
            if k not in ['id', 'fileanme', 'width', 'height', 'ann']:
                # 
                patch_info[k] = v

        window = windows[i]
        x_start, y_start, x_stop, y_stop = window.tolist()
        patch_info['x_start'] = x_start
        patch_info['y_start'] = y_start

        # 
        patch = img[:, y_start:y_stop, x_start:x_stop]

        if not no_padding:
            height = y_stop - y_start
            width = x_stop - x_start

            if height > patch.shape[1] or width > patch.shape[2]:
            # if height > patch.shape[0] or width > patch.shape[1]:
                padding_patch = np.empty((height, width,patch.shape[0]),
                                         dtype=np.float32)
                if not isinstance(padding_value, (int, float)):
                    # print('patch.shape',patch.shape)[3, 800, 800]
                    assert len(padding_value) == patch.shape[0]
                padding_patch[...] = padding_value# 800,800,3)
                # (3,800,800)
                padding_patch=torch.tensor(padding_patch.transpose((2,1,0)),
                                           device=patch.device)
                padding_patch[ ...,:patch.shape[1], :patch.shape[2]] = patch
                patch = padding_patch
        patch_info['height'] = patch.shape[1]
        patch_info['width'] = patch.shape[2]

        patch_infos.append(patch_info)
        patchs.append(patch)

    return patchs, patch_infos


def translate(bboxes, x, y):
    """Map bboxes from window coordinate back to original coordinate.

    Args:
        bboxes (np.array): bboxes with window coordinate.
        x (float): Deviation value of x-axis.
        y (float): Deviation value of y-axis

    Returns:
        np.array: bboxes with original coordinate.
    """
    dim = bboxes.shape[-1]
    translated = bboxes + np.array([x, y] * int(dim / 2), dtype=np.float32)
    return translated

# def translate_bboxes(bboxes, offset):
#     """Translate bboxes according to its shape.
#
#     If the bbox shape is (n, 5), the bboxes are regarded as horizontal bboxes
#     and in (x, y, x, y, score) format. If the bbox shape is (n, 6), the bboxes
#     are regarded as rotated bboxes and in (x, y, w, h, theta, score) format.
#
#     Args:
#         bboxes (np.ndarray): The bboxes need to be translated. Its shape can
#             only be (n, 5) and (n, 6).
#         offset (np.ndarray): The offset to translate with shape being (2, ).
#
#     Returns:
#         np.ndarray: Translated bboxes.
#     """
#     if bboxes.shape[1] == 5:
#         bboxes[:, :4] = bboxes[:, :4] + np.tile(offset, 2)
#     elif bboxes.shape[1] == 6:
#         bboxes[:, :2] = bboxes[:, :2] + offset
#     else:
#         raise TypeError('Require the shape of `bboxes` to be (n, 5) or (n, 6),'
#                         f' but get `bboxes` with shape being {bboxes.shape}.')
#     return bboxes

def list2tensor_(img_lists ,dim=0):
    '''
    images: list of list of tensor images
    '''
    if len(img_lists)>0:
        device = img_lists[0].device
        inputs = torch.cat([img_list for img_list in img_lists], dim=dim).to(device)
    else:
        inputs = torch.tensor([])

    return inputs

def merge_results(results, iou_thr=0.4):
    """Merge patch results via nms.

    Args:
        results (list[tensor] | list[tuple]): A list of patches results.
        offsets (np.ndarray): Positions of the left top points of patches.
        img_shape (tuple): A tuple of the huge image's width and height.
        iou_thr (float): The IoU threshold of NMS.
        device (str): The device to call nms.

    Retunrns:
        list[np.ndarray]: Detection results after merging.
    """

    merge_results = []
    for ls in results:
        if isinstance(ls,list):
            for idx in range(len(ls)):
                patches = ls[idx]
                if isinstance(patches, list):
                    for patch in patches:
                        merge_results.append(patch)
                else:
                    merge_results.append(patches)
        elif isinstance(ls, tuple):
            merge_results.append(ls)
        else:
            merge_results.append(ls)

    num_patches = len(merge_results)
    num_classes = 1
    nms_func = nms

    merged_bboxes = []
    merged_labels=[]
    bbox_list=[]

    for i in range(num_patches):
        p_list = list(merge_results[i])
        # get boxxes and labels
        dets_per_cls = p_list[0]
        labels_per_cls = p_list[1]

        if dets_per_cls.size()[0] == 0:
            merged_bboxes.append(dets_per_cls)
            # if with_mask:
            #     merged_masks.append(masks_per_cls)
        else:

            nms_dets, keeps = nms_func(dets_per_cls[:, :-1],
                                       dets_per_cls[:, -1], iou_thr)
            nms_labels = labels_per_cls[keeps]

            merged_bboxes.append(nms_dets)
            merged_labels.append(nms_labels)

    out_bboxes = list2tensor_(merged_bboxes, dim=0)
    out_labels = list2tensor_(merged_labels, dim=0)
    # NMS to all boxxes(from local and global)
    out_bboxes, keeps_out = nms_func(out_bboxes[:, :-1],
                                       out_bboxes[:, -1], iou_thr)

    # if out_bboxes.shape[0]>max_bbox_num:
    #     out_bboxes=out_bboxes[0:max_bbox_num]
    #     keeps_out=keeps_out[0:max_bbox_num]

    out_labels = out_labels[keeps_out]

    return out_bboxes, out_labels

def merge_results_two_stage_hbb_danleiyuan(local_bboxes_lists, iou_thr=0.4):
    """Merge patch results via nms.

    Args:
        local_bboxes_lists : list[ndarray[]].
        offsets (np.ndarray): Positions of the left top points of patches.
        img_shape (tuple): A tuple of the huge image's width and height.
        iou_thr (float): The IoU threshold of NMS.
        device (str): The device to call nms.

    Retunrns:
        list[np.ndarray]: Detection results after merging.
    """

    merge_results = []
    results=local_bboxes_lists
    for ls in results:
        if isinstance(ls,list):
            for idx in range(len(ls)): #               
                patches = ls[idx]
                if isinstance(patches, list):
                    patch=torch.tensor(patches)
                    merge_results.append(patch)
                    # for patch in patches:
                    #     merge_results.append(patch)
                else:
                    merge_results.append(torch.tensor(patches))
        # elif isinstance(ls, tuple):
        #     merge_results.append(ls)


    num_patches = len(merge_results)
    num_classes = 1
    nms_func = nms
    merged_bboxes = []

    for i in range(num_patches):
        p_list = list(merge_results[i])
        # get boxxes and labels
        dets_per_cls = p_list[0]

        if dets_per_cls.size()[0] == 0:
            merged_bboxes.append(dets_per_cls)
            # if with_mask:
            #     merged_masks.append(masks_per_cls)
        else:

            nms_dets, keeps = nms_func(dets_per_cls[:, :-1],
                                       dets_per_cls[:, -1], iou_thr)
            # nms_labels = labels_per_cls[keeps]

            merged_bboxes.append(nms_dets)
            # merged_labels.append(nms_labels)

    out_bboxes = list2tensor_(merged_bboxes, dim=0)
    # out_labels = list2tensor_(merged_labels, dim=0)
    # NMS to all boxxes(from local and global)
    # print('out_bboxes shape:',out_bboxes.shape)
    if out_bboxes.shape[0]!=0:
        out_bboxes, keeps_out = nms_func(out_bboxes[:, :-1],
                                     out_bboxes[:, -1], iou_thr)
    else:
        out_bboxes= torch.zeros((0, 5), device=out_bboxes.device)

    # if out_bboxes.shape[0]>max_bbox_num:
    #     out_bboxes=out_bboxes[0:max_bbox_num]
    #     keeps_out=keeps_out[0:max_bbox_num]

    # out_labels = out_labels[keeps_out]

    return out_bboxes

def merge_results_two_stage_hbb(local_bboxes_lists, iou_thr=0.4,flag = None):
    merge_results = []

    if flag == 1:
        results = []
        for local_tem in local_bboxes_lists:
            results = results + local_tem
        # results=local_bboxes_lists[0]

        #### 
        all_patch = results
    elif flag == 2:
        all_patch = []
        results = local_bboxes_lists
        len_result = len(results)
        for r in range(len_result):
            all_patch = all_patch + results[r]
    else:
        all_patch = local_bboxes_lists
       


    nms_func = nms # for HBB
    merged_bboxes = []
    class_keep = []
    for class_id in  range(len(all_patch[0])):
        # class_keep.append([])
        cla_patch = []
        for p_id in range(len(all_patch)):
            if len(all_patch[p_id][class_id]) == 0:
                continue
            else:
                cla_patch.append(all_patch[p_id][class_id])
        if len(cla_patch) != 0:
            me_cla_patch = np.concatenate(cla_patch, axis=0)  ### num,6
            dets_per_cls = torch.from_numpy(me_cla_patch)
            nms_dets, keeps = nms_func(dets_per_cls[:, :-1],
                                        dets_per_cls[:, -1], iou_thr)
            class_keep.append(keeps)
            merged_bboxes.append(nms_dets.cpu().numpy())
        else:
            merged_bboxes.append(torch.zeros((0, 5)).cpu().numpy())
            class_keep.append([])
            continue
    
    all_class_keep = []
    ### all classes to nms_rotated
    final_merged_bboxes = []
    for meb in merged_bboxes:
         tensor_meb = torch.from_numpy(meb)
         fin_nms_dets, keeps = nms_func(tensor_meb[:, :-1],
                                       tensor_meb[:, -1], iou_thr)
         all_class_keep.append(keeps)
         final_merged_bboxes.append(fin_nms_dets.cpu().numpy())

    if (flag == 1) or (flag == 2) or (flag == 3):
           #
        #    new = []
        #    for ck1,ck2 in zip(class_keep,all_class_keep):
        #        if len(ck1) == 0:
        #            new.append([])
        #        else:
        #            new.append(ck1[ck2])
           return final_merged_bboxes #,new
                   
    # if flag == 1:
    #        return final_merged_bboxes
    else:
           return final_merged_bboxes

def FullImageCrop(self, imgs, bboxes, labels, patch_shape,
                  gaps,
                  jump_empty_patch=False,
                  mode='train'):
    """
    Args:
        imgs (Tensor): Input images of shape (N, C, H, W).
            Typically these should be mean centered and std scaled.
        bboxes (list[Tensor]): Each item are the truth boxes for each
            image in [tl_x, tl_y, br_x, br_y] format.
        labels (list[Tensor]): Class indices corresponding to each box
    Returns:
        dict[str, Tensor]: A dictionary of loss components.
    """
    out_imgs = []
    out_bboxes = []
    out_labels = []
    out_metas = []
    device = get_device()
    img_rate_thr = 0.6  # 
    iof_thr = 0.1  #
    if mode == 'train':
        # for i in range(imgs.shape[0]):
        for img, bbox, label in zip(imgs, bboxes, labels):
            p_imgs = []
            p_bboxes = []
            p_labels = []
            p_metas = []
            img = img.cpu()
            # patch
            info = dict()
            info['labels'] = np.array(torch.tensor(label, device='cpu', requires_grad=False))
            info['ann'] = {'bboxes': {}}
            info['width'] = img.shape[1]
            info['height'] = img.shape[2]

            tmp_boxes = torch.tensor(bbox, device='cpu', requires_grad=False)
            info['ann']['bboxes'] = np.array(hbb2obb(tmp_boxes, 'oc'))  #             
            info['ann']['bboxes'] = np.array(obb2poly_oc(tmp_boxes))  # 
            bbbox = info['ann']['bboxes']
            # sizes = [patch_shape[0]]
            sizes = [128]
            # gaps=[0]
            windows = get_sliding_window(info, sizes, gaps, img_rate_thr)
            window_anns = get_window_obj(info, windows, iof_thr)
            patchs, patch_infos = crop_and_save_img(info, windows, window_anns,
                                                    img,
                                                    no_padding=True,
                                                    # no_padding=False,
                                                    padding_value=[104, 116, 124])

            #            
            for i, patch_info in enumerate(patch_infos):
                if jump_empty_patch:
                    # 
                    if patch_info['labels'] == [-1]:
                        # print('Patch does not contain box.\n')
                        continue
                obj = patch_info['ann']
                if min(obj['bboxes'].shape) == 0:  # 
                    tmp_boxes = poly2obb_oc(torch.tensor(obj['bboxes']))  #                     
                    tmp_boxes=obb2hbb(tmp_boxes,'oc')
                else:
                    tmp_boxes = poly2obb_oc(torch.tensor(obj['bboxes']))  # 
                    tmp_boxes = obb2hbb(tmp_boxes, 'oc')
                p_bboxes.append(tmp_boxes.to(device))
                # p_trunc.append(torch.tensor(obj['trunc'],device=device))  # 
                ## 
                p_labels.append(torch.tensor(patch_info['labels'], device=device))
                p_metas.append({'x_start': torch.tensor(patch_info['x_start'], device=device),
                                'y_start': torch.tensor(patch_info['y_start'], device=device),
                                'ori_shape': patch_shape,'shape': patch_shape, 
                                'trunc': torch.tensor(obj['trunc'], device=device)})

                patch = patchs[i]
                p_imgs.append(patch.to(device))

            out_imgs.append(p_imgs)
            out_bboxes.append(p_bboxes)
            out_labels.append(p_labels)
            out_metas.append(p_metas)

    elif mode == 'test':
        p_imgs = []
        p_metas = []
        img = imgs.cpu().squeeze(0)
        # patch
        info = dict()
        info['labels'] = np.array(torch.tensor([], device='cpu'))
        info['ann'] = {'bboxes': {}}
        info['width'] = img.shape[1]
        info['height'] = img.shape[2]

        sizes = [patch_shape[0]]
        # gaps=[0]
        windows = get_sliding_window(info, sizes, gaps, img_rate_thr)
        patchs, patch_infos = crop_img_withoutann(info, windows, img,
                                                  no_padding=False,
                                                  padding_value=[104, 116, 124])

        #      
        for i, patch_info in enumerate(patch_infos):
            p_metas.append({'x_start': torch.tensor(patch_info['x_start'], device=device),
                            'y_start': torch.tensor(patch_info['y_start'], device=device),
                            'ori_shape': patch_shape,
                            'shape': patch_shape, 'img_shape': patch_shape, 'scale_factor': 1})

            patch = patchs[i]
            p_imgs.append(patch.to(device))

        out_imgs.append(p_imgs)
        out_metas.append(p_metas)

        return out_imgs, out_metas

    return out_imgs, out_bboxes, out_labels, out_metas

def merge_results_tensor_hbb(all_bboxes_list, iou_thr=0.4):

    out_bboxes = list2tensor_(all_bboxes_list, dim=0)
    nms_func = nms
    # NMS to all boxxes(from local and global)
    out_bboxes, keeps_out = nms_func(out_bboxes[:, :-1],
                                     out_bboxes[:, -1], iou_thr)

    # if out_bboxes.shape[0]>max_bbox_num:
    #     out_bboxes=out_bboxes[0:max_bbox_num]
    #     keeps_out=keeps_out[0:max_bbox_num]
    # out_labels = out_labels[keeps_out]
    return out_bboxes