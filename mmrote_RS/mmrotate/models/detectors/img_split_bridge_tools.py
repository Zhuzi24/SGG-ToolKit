# Copyright (c) OpenMMLab. All rights reserved.
# Reference: https://github.com/jbwang1997/BboxToolkit

import itertools
from math import ceil
from multiprocessing import Manager, Pool
from mmrotate.core import (build_assigner, build_sampler,rbbox2result,
                           multiclass_nms_rotated, obb2poly, poly2obb)
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
    img = img.clone()
    patchs=[]
    patch_infos = []
    for i in range(windows.shape[0]):
        patch_info = dict()
        for k, v in info.items():
            if k not in ['id', 'fileanme', 'width', 'height', 'ann']:
                # 将不在上述5个键里的键值提取出来,这里将labels提取
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

        # 可以用float32以及更低的数字?因为都是对归一化图进行操作
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
                padding_patch[...] = padding_value# 此时维度为(800,800,3)
                # 再交换到正确维度(3,800,800)
                padding_patch=torch.tensor(padding_patch.transpose((2,1,0)),
                                           device=patch.device)
                padding_patch[ ...,:patch.shape[1], :patch.shape[2]] = patch
                patch = padding_patch
        patch_info['height'] = patch.shape[1]
        patch_info['width'] = patch.shape[2]

        bboxes_num = patch_info['ann']['bboxes'].shape[0]
        # outdir = os.path.join(anno_dir, patch_info['id'] + '.txt')
        #
        # with codecs.open(outdir, 'w', 'utf-8') as f_out:
        # patch_info['labels'] =[]
        patch_label = []

        if bboxes_num == 0:
            # patch_info['labels']=[-1]  # 背景类别
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
                # 将不在上述5个键里的键值提取出来,这里将labels提取
                patch_info[k] = v

        window = windows[i]
        x_start, y_start, x_stop, y_stop = window.tolist()
        patch_info['x_start'] = x_start
        patch_info['y_start'] = y_start

        # 可以用float32以及更低的数字?因为都是对归一化图进行操作
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
                padding_patch[...] = padding_value# 此时维度为(800,800,3)
                # 再交换到正确维度(3,800,800)
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
    if not torch.is_tensor(img_lists[1]): 
        img_lists[1]= torch.cat([v if torch.is_tensor(v)  else torch.tensor(v) for v in img_lists[1]])
    if len(img_lists)>0:
        device = img_lists[0].device
        inputs = torch.cat([img_list.to(device) for img_list in img_lists], dim=dim)
    else:
        inputs = torch.tensor([])
        # inputs = torch.tensor([],device=device)

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

    num_patches = len(merge_results)
    num_classes = 1
    nms_func = nms_rotated

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
    nms_func = nms_rotated
    # NMS to all boxxes(from local and global)
    out_bboxes, keeps_out = nms_func(out_bboxes[:, :-1],
                                       out_bboxes[:, -1], iou_thr)

    # if out_bboxes.shape[0]>max_bbox_num:
    #     out_bboxes=out_bboxes[0:max_bbox_num]
    #     keeps_out=keeps_out[0:max_bbox_num]

    out_labels = out_labels[keeps_out]

    return out_bboxes, out_labels

# def merge_results_two_stage(local_bboxes_lists, iou_thr=0.4): # TODO: 这个是为了DOTA设计的,定义了class 15类
#     """Merge patch results via nms.

#     Args:
#         results (list[tensor] | list[tuple]): A list of patches results.
#         offsets (np.ndarray): Positions of the left top points of patches.
#         img_shape (tuple): A tuple of the huge image's width and height.
#         iou_thr (float): The IoU threshold of NMS.
#         device (str): The device to call nms.

#     Retunrns:
#         list[np.ndarray]: Detection results after merging.
#     """

#     merge_results = []
#     results=local_bboxes_lists
#     for ls in results:
#         if isinstance(ls,list):
#             for idx in range(len(ls)): # 每个patch的结果
#                 patches = ls[idx]
#                 if isinstance(patches, list):
#                     # patch=torch.tensor(np.array(patches))
#                     merge_results.append(patches)
#                     # for patch in patches:
#                     #     merge_results.append(patch)
#                 else:
#                     # merge_results.append(torch.tensor(patches))
#                     merge_results.append(patches)

#     num_patches = len(merge_results)
    # # num_classes = 15
    # num_classes = 1
#     nms_func = nms_rotated

    
#     out_bboxes_all=[]

#     for cls in range(num_classes):
#         merged_bboxes = []
#         for i in range(num_patches):
#             p_list = list(merge_results[i][cls])
#             if len(p_list)==0:
#                 merged_bboxes.append(torch.tensor(merge_results[i][cls]))
#             else:
#                 p_list =p_list[0]
#                 p_list = torch.tensor(np.expand_dims(p_list, axis=0))
#                 nms_dets, keeps = nms_func(p_list[:, :-1],
#                                            p_list[:, -1], iou_thr)
                
#                 merged_bboxes.append(nms_dets)
#                 # merged_labels.append(nms_labels)

#         out_bboxes = list2tensor_(merged_bboxes, dim=0) # torch.size[3648,6]
#         # out_labels = list2tensor_(merged_labels, dim=0)
#         nms_func = nms_rotated
#         # NMS to all boxxes(from local and global)
#         out_bboxes_single_cls, keeps_out = nms_func(out_bboxes[:, :-1],
#                                             out_bboxes[:, -1], iou_thr)
#         if out_bboxes_single_cls.size()[1]==5:
#             out_bboxes_single_cls = torch.cat((out_bboxes_single_cls, torch.empty(0, 1)), dim=1)

#         out_bboxes_single_cls=out_bboxes_single_cls.numpy()
#         out_bboxes_all.append(out_bboxes_single_cls)

#     return out_bboxes_all

# def merge_results_two_stage(local_bboxes_lists, iou_thr=0.4):
#     merge_results = []
#     results=local_bboxes_lists
#     for ls in results:
#         if isinstance(ls,list):
#             for idx in range(len(ls)): # 每个patch的结果
#                 patches = ls[idx]
#                 if isinstance(patches, list):
#                     patch=torch.tensor(np.array(patches))
#                     merge_results.append(patch)
#                     # for patch in patches:
#                     #     merge_results.append(patch)
#                 else:
#                     merge_results.append(torch.tensor(patches))
#         # elif isinstance(ls, tuple):
#         #     merge_results.append(ls)
#     num_patches = len(merge_results)
#     nms_func = nms_rotated
#     merged_bboxes = []
#     for i in range(num_patches):
#         p_list = list(merge_results[i])
#         dets_per_cls = p_list[0]

#         if dets_per_cls.size()[0] == 0:
#             merged_bboxes.append(dets_per_cls)
#         else:
#             nms_dets, keeps = nms_func(dets_per_cls[:, :-1],
#                                        dets_per_cls[:, -1], iou_thr)
#             merged_bboxes.append(nms_dets)
#     out_bboxes = list2tensor_(merged_bboxes, dim=0)
#     nms_func = nms_rotated
#     out_bboxes, keeps_out = nms_func(out_bboxes[:, :-1],
#                                        out_bboxes[:, -1], iou_thr)
    

#     return out_bboxes

# def merge_results_two_stage(local_bboxes_lists, iou_thr=0.4, flag=None):
#     merge_results = []
#     nms_func = nms_rotated

#     if flag in {1, 2}:
#         all_patch = [item for sublist in local_bboxes_lists for item in sublist]
#     else:
#         all_patch = local_bboxes_lists

#     merged_bboxes = []
#     class_keep = []

#     for class_id in range(len(all_patch[0])):
#         cla_patch = [all_patch[p_id][class_id] for p_id in range(len(all_patch)) if len(all_patch[p_id][class_id]) != 0]

#         if cla_patch:
#             me_cla_patch = np.concatenate(cla_patch, axis=0)  ### num,6
#             dets_per_cls = torch.from_numpy(me_cla_patch)
#             nms_dets, keeps = nms_func(dets_per_cls[:, :-1], dets_per_cls[:, -1], iou_thr)
#             class_keep.append(keeps)
#             merged_bboxes.append(nms_dets.cpu().numpy())
#         else:
#             merged_bboxes.append(np.zeros((0, 6)))
#             class_keep.append([])

#     all_class_keep = []
#     final_merged_bboxes = []

#     for meb in merged_bboxes:
#         tensor_meb = torch.from_numpy(meb)
#         fin_nms_dets, keeps = nms_func(tensor_meb[:, :-1], tensor_meb[:, -1], iou_thr)
#         all_class_keep.append(keeps)
#         final_merged_bboxes.append(fin_nms_dets.cpu().numpy())

#     if flag in {1, 2, 3}:
#         new = [ck1[ck2] if ck1 else [] for ck1, ck2 in zip(class_keep, all_class_keep)]
#         return final_merged_bboxes, new

#     return final_merged_bboxes




def merge_results_two_stage(local_bboxes_lists, iou_thr=0.4,flag = None):
    merge_results = []

    if flag == 1:
        results = []
        for local_tem in local_bboxes_lists:
            results = results + local_tem
        # results=local_bboxes_lists[0]

        #### 获取所有patch
        all_patch = results
    elif flag == 2:
        all_patch = []
        results = local_bboxes_lists
        len_result = len(results)
        for r in range(len_result):
            all_patch = all_patch + results[r]
    else:
        all_patch = local_bboxes_lists
       


    nms_func = nms_rotated
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
            merged_bboxes.append(torch.zeros((0, 6)).cpu().numpy())
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
           new = []
           for ck1,ck2 in zip(class_keep,all_class_keep):
               if len(ck1) == 0:
                   new.append([])
               else:
                   new.append(ck1[ck2])
           return final_merged_bboxes,new
                   
    # if flag == 1:
    #        return final_merged_bboxes
    else:
           return final_merged_bboxes



def merge_results_two_stage_2model_TTA(local_bboxes_lists, global_bboxes_list_ori,global_bbox_list_2, iou_thr=0.4):
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

    # 对local box添加权重
    # weight1=1
    # weight2=0
    # weight_local = weight1/(weight1+weight2)
    # weight_global = weight2/(weight1+weight2)
    weight_local =1
    weight_global =1

    merge_results = []
    results=local_bboxes_lists
    for ls in results:
        if isinstance(ls,list):
            for idx in range(len(ls)): # 每个patch的结果
                patches = ls[idx]
                if isinstance(patches, list):
                    patch=torch.tensor(np.array(patches))
                    patch[:,:,-1]=patch[:,:,-1]*weight_local  # score加权
                    merge_results.append(patch)
                else:
                    tmp_patch=torch.tensor(patches)
                    tmp_patch[:,:, -1] = tmp_patch[:,:, -1] * weight_local  # score加权
                    merge_results.append(tmp_patch)


    # nms前根据预测box大小进行筛选
    # high_box=[0.97,0.98,0.99]
    global_bboxes_list = np.array(global_bboxes_list_ori[0]).squeeze(0)
    delta = 0
    for arr in range(len(global_bboxes_list)):
        arr2 = arr-delta
        tmp_gbox = global_bboxes_list[arr2, :]
        # if tmp_gbox[2] < 100 and tmp_gbox[3] < 100 and tmp_gbox[-1]<0.2:
        if tmp_gbox[2]*tmp_gbox[3] < 4096:
            # global_bboxes_list[arr][-1] = 0.001
            global_bboxes_list = np.delete(global_bboxes_list, arr2, 0)
            delta+=1
        if tmp_gbox[-1] < 0.05:
            # global_bboxes_list[arr][-1] = 0.001
            global_bboxes_list = np.delete(global_bboxes_list, arr2, 0)
            delta+=1
        # else:
        #     global_bboxes_list[arr2][-1]=high_box[np.int(arr2%3)]
    
    print('delta:',delta)

    global_bboxes_list[:,-1]=global_bboxes_list[:,-1]*weight_global
    merge_results.append(torch.tensor(global_bboxes_list).unsqueeze(0))


    global_bboxes_list_2= np.array(global_bbox_list_2[0]).squeeze(0)
    delta2 = 0
    for arr in range(len(global_bboxes_list_2)):
        arr2 = arr-delta2
        tmp_gbox = global_bboxes_list_2[arr2, :]
        # if tmp_gbox[2] < 100 and tmp_gbox[3] < 100 and tmp_gbox[-1]<0.2:
        if tmp_gbox[2]*tmp_gbox[3] < 4096:
            # global_bboxes_list[arr][-1] = 0.001
            global_bboxes_list_2 = np.delete(global_bboxes_list_2, arr2, 0)
            delta2+=1
        # if tmp_gbox[-1] < 0.05:
        #     global_bboxes_list_2 = np.delete(global_bboxes_list_2, arr2, 0)
        #     delta2+=1
        # else:
        #     global_bboxes_list[arr2][-1]=high_box[np.int(arr2%3)]
    
    print('delta2:',delta2)

    global_bboxes_list_2[:,-1]=global_bboxes_list_2[:,-1]*weight_global
    merge_results.append(torch.tensor(global_bboxes_list_2).unsqueeze(0))



    num_patches = len(merge_results)
    num_classes = 1
    nms_func = nms_rotated

    merged_bboxes = []
    merged_labels=[]
    bbox_list=[]

    for i in range(num_patches):
        p_list = list(merge_results[i])
        # get boxxes and labels
        dets_per_cls = p_list[0]
        # labels_per_cls = p_list[1]

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
    nms_func = nms_rotated
    # NMS to all boxxes(from local and global)
    out_bboxes, keeps_out = nms_func(out_bboxes[:, :-1],
                                       out_bboxes[:, -1], iou_thr)

    return out_bboxes



def merge_results_two_stage_2model(local_bboxes_lists, global_bboxes_list_ori, iou_thr=0.2):
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

    # 对local box添加权重
    # weight1=1
    # weight2=0
    # weight_local = weight1/(weight1+weight2)
    # weight_global = weight2/(weight1+weight2)
    weight_local =1
    weight_global =1

    merge_results = []
    results=local_bboxes_lists
    for ls in results:
        if isinstance(ls,list):
            for idx in range(len(ls)): # 每个patch的结果
                patches = ls[idx]
                if isinstance(patches, list):
                    patch=torch.tensor(np.array(patches))
                    patch[:,:,-1]=patch[:,:,-1]*weight_local  # score加权
                    merge_results.append(patch)
                else:
                    tmp_patch=torch.tensor(patches)
                    tmp_patch[:,:, -1] = tmp_patch[:,:, -1] * weight_local  # score加权
                    merge_results.append(tmp_patch)


    # nms前根据预测box大小进行筛选
    # high_box=[0.97,0.98,0.99]
    global_bboxes_list = np.array(global_bboxes_list_ori[0]).squeeze(0)
    delta = 0
    for arr in range(len(global_bboxes_list)):
        arr2 = arr-delta
        tmp_gbox = global_bboxes_list[arr2, :]
        # if tmp_gbox[2] < 100 and tmp_gbox[3] < 100 and tmp_gbox[-1]<0.2:
        if tmp_gbox[2]*tmp_gbox[3] < 4096:
            # global_bboxes_list[arr][-1] = 0.001
            global_bboxes_list = np.delete(global_bboxes_list, arr2, 0)
            delta+=1
        # if tmp_gbox[-1] < 0.05:
        #     # global_bboxes_list[arr][-1] = 0.001
        #     global_bboxes_list = np.delete(global_bboxes_list, arr2, 0)
        #     delta+=1
        # else:
        #     global_bboxes_list[arr2][-1]=high_box[np.int(arr2%3)]
    
    print('delta:',delta)

    global_bboxes_list[:,-1]=global_bboxes_list[:,-1]*weight_global
    merge_results.append(torch.tensor(global_bboxes_list).unsqueeze(0))

    num_patches = len(merge_results)
    num_classes = 1
    nms_func = nms_rotated

    merged_bboxes = []
    merged_labels=[]
    bbox_list=[]

    for i in range(num_patches):
        p_list = list(merge_results[i])
        # get boxxes and labels
        dets_per_cls = p_list[0]
        # labels_per_cls = p_list[1]

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
    nms_func = nms_rotated
    # NMS to all boxxes(from local and global)
    out_bboxes, keeps_out = nms_func(out_bboxes[:, :-1],
                                       out_bboxes[:, -1], iou_thr)

    return out_bboxes

def merge_results_tensor(all_bboxes_list, iou_thr=0.4):

    out_bboxes = list2tensor_(all_bboxes_list, dim=0)
    nms_func = nms_rotated
    # NMS to all boxxes(from local and global)
    out_bboxes, keeps_out = nms_func(out_bboxes[:, :-1],
                                     out_bboxes[:, -1], iou_thr)

    # if out_bboxes.shape[0]>max_bbox_num:
    #     out_bboxes=out_bboxes[0:max_bbox_num]
    #     keeps_out=keeps_out[0:max_bbox_num]
    # out_labels = out_labels[keeps_out]
    return out_bboxes



# 加权融合
def merge_results_tensor_global_local(global_bboxes_list, local_bboxes_list, iou_thr=0.4):
    global_bboxes = list2tensor_(global_bboxes_list, dim=0)
    local_bboxes= list2tensor_(local_bboxes_list, dim=0)

    weight_local =1
    weight_global =1
    
    # TODO:将global_boxes中的小型桥梁置信度降低
    # if tmp_gbox[-1] < 0.05:
    #     # global_bboxes_list[arr][-1] = 0.001
    #     global_bboxes_list = np.delete(global_bboxes_list, arr2, 0)
    #     delta+=1
    # else:
    #     global_bboxes_list[arr2][-1]=high_box[np.int(arr2%3)]




    nms_func = nms_rotated
    out_bboxes=torch.cat(global_bboxes,local_bboxes,dim=0)
    # NMS to all boxxes(from local and global)
    out_bboxes, keeps_out = nms_func(out_bboxes[:, :-1],
                                     out_bboxes[:, -1], iou_thr)


    # if out_bboxes.shape[0]>max_bbox_num:
    #     out_bboxes=out_bboxes[0:max_bbox_num]
    #     keeps_out=keeps_out[0:max_bbox_num]
    # out_labels = out_labels[keeps_out]
    return out_bboxes