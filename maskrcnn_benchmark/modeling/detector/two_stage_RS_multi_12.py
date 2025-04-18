# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from mmcv.ops import box_iou_rotated
import torch
from torch import nn
from mmcv.cnn import ConvModule
from mmrotate.core import (build_assigner, build_sampler,rbbox2result,
                           multiclass_nms_rotated, obb2poly, poly2obb)
from mmrotate.models.builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from mmrotate.models.detectors.base import RotatedBaseDetector
import numpy as np
from mmrotate.models.detectors.img_split_bridge_tools import *
from mmdet.utils import get_device
from PIL import Image
import torch.nn.functional as F
import mmcv
from pathlib import Path
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmdet.core import get_classes
from mmrotate.apis import inference_detector_by_patches
from maskrcnn_benchmark.modeling.detector.base_RS import BaseDetector
from maskrcnn_benchmark.structures.image_list import to_image_list
from mmdet.datasets.pipelines.transforms import Resize
import copy
from mmcv.ops import box_iou_rotated

from copy import deepcopy

POS = [48, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
class RResize(Resize):
    """Resize images & rotated bbox Inherit Resize pipeline class to handle
    rotated bboxes.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio).
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None):
        super(RResize, self).__init__(
            img_scale=img_scale,
            multiscale_mode=multiscale_mode,
            ratio_range=ratio_range,
            keep_ratio=True)

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in results.get('bbox_fields', []):
            bboxes = results[key]
            orig_shape = bboxes.shape
            bboxes = bboxes.reshape((-1, 5))
            w_scale, h_scale, _, _ = results['scale_factor']
            bboxes[:, 0] *= w_scale
            bboxes[:, 1] *= h_scale
            bboxes[:, 2:4] *= np.sqrt(w_scale * h_scale)
            results[key] = bboxes.reshape(orig_shape)

RZ = RResize(img_scale=(1024,1024))

def hs(pathches_cls_scores, p_keeps):
    all_cls = [item for sublist in pathches_cls_scores for item in sublist]
    
    mer_cls_scores = []
    for cls_id in np.arange(48):
        cls_part = [all_cls[pp][cls_id].reshape((1, 49)) if all_cls[pp][cls_id].ndim == 1 
                    else all_cls[pp][cls_id].reshape((1,49,5)) if all_cls[pp][cls_id].shape == (49, 5) 
                    else all_cls[pp][cls_id] 
                    for pp in np.arange(len(all_cls)) if len(all_cls[pp][cls_id]) != 0]
        mer_cls_scores.append(np.concatenate(cls_part, axis=0) if cls_part else [])

    # new_mer_cls_scores = [np.expand_dims(ck1[ck2], axis=0) if len(ck2) == 1 else ck1[ck2] 
    #                       for ck1, ck2 in zip(mer_cls_scores, p_keeps) if len(ck1)]
        
    new_mer_cls_scores = []
    for ck1,ck2 in zip(mer_cls_scores,p_keeps):
        if len(ck1) == 0:
            new_mer_cls_scores.append([])
        else:
            if len(ck2) == 1:
                new_mer_cls_scores.append(np.expand_dims(ck1[ck2], axis=0))
            else:
                new_mer_cls_scores.append(ck1[ck2])

    return new_mer_cls_scores



def hs_all(all_scores, all_keeps):
    mer_cls_scores = []
    for cls_id in np.arange(48):
        cls_part = [all_scores[pp][cls_id].reshape((1, 49)) if all_scores[pp][cls_id].ndim == 1 
                    else all_scores[pp][cls_id].reshape((1,49,5)) if all_scores[pp][cls_id].shape == (49, 5) 
                    else all_scores[pp][cls_id] 
                    for pp in np.arange(len(all_scores)) if len(all_scores[pp][cls_id]) != 0]
        mer_cls_scores.append(np.concatenate(cls_part, axis=0) if cls_part else [])


    new_mer_cls_scores = []
    for ck1,ck2 in zip(mer_cls_scores,all_keeps):
        if len(ck1) == 0:
            new_mer_cls_scores.append([])
        else:
            if len(ck2) == 1:
                new_mer_cls_scores.append(np.expand_dims(ck1[ck2], axis=0))
            else:
                new_mer_cls_scores.append(ck1[ck2])
    
    return new_mer_cls_scores


    # new_mer_cls_scores = [np.expand_dims(ck1[ck2], axis=0) if len(ck2) == 1 else ck1[ck2] 
    #                       for ck1, ck2 in zip(mer_cls_scores, all_keeps) if ck1]

    # return new_mer_cls_scores


 
def resize_bboxes_len6(bboxes_out,each_class,scale):
    """Resize bounding boxes with scales."""

    for i in range(len(bboxes_out)):
        box_out=bboxes_out[i]
        each_class_tmp = each_class[i]
        w_scale = scale
        h_scale = scale
        box_out[:, 0] *= w_scale
        box_out[:, 1] *= h_scale
        box_out[:, 2:4] *= np.sqrt(w_scale * h_scale)

        if len(each_class_tmp) != 0:
            each_class_tmp[:,:, 0] *= w_scale
            each_class_tmp[:,:, 1] *= h_scale
            each_class_tmp[:,:, 2:4] *= np.sqrt(w_scale * h_scale)


    return bboxes_out



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
    img_rate_thr = 0.6  # 图片与wins窗口的交并比阈值
    iof_thr = 0.1  # 裁剪后的标签占原标签的比值阈值
    padding_value = [0.0081917211329, -0.004901960784, 0.0055655449953]  # 归一化后的padding值

    # if mode == 'train':
    #     # for i in range(imgs.shape[0]):
    #     for img, bbox, label in zip(imgs, [bboxes], [labels]):
    #         p_imgs = []
    #         p_bboxes = []
    #         p_labels = []
    #         p_metas = []
    #         img = img.cpu()
    #         # patch
    #         info = dict()
    #         info['labels'] = np.array(torch.tensor(label, device='cpu', requires_grad=False))
    #         info['ann'] = {'bboxes': {}}
    #         info['width'] = img.shape[2]
    #         info['height'] = img.shape[1]

    #         tmp_boxes = torch.tensor(bbox, device='cpu', requires_grad=False)
    #         info['ann']['bboxes'] = np.array(obb2poly(tmp_boxes, self.version))  # 这里将OBB转换为8点表示形式
    #         bbbox = info['ann']['bboxes']
    #         sizes = [patch_shape[0]]
    #         # gaps=[0]
    #         windows = get_sliding_window(info, sizes, gaps, img_rate_thr)
    #         window_anns = get_window_obj(info, windows, iof_thr)
    #         patchs, patch_infos = crop_and_save_img(info, windows, window_anns,
    #                                                 img,
    #                                                 no_padding=True,
    #                                                 # no_padding=False,
    #                                                 padding_value=padding_value)

    #         # 对每张大图分解成的子图集合中的每张子图遍历
    #         for i, patch_info in enumerate(patch_infos):
    #             if jump_empty_patch:
    #                 # 如果该patch中不含有效标签,将其跳过不输出,可在训练时使用

    #                 if patch_info['labels'] == [-1]:
    #                     # print('Patch does not contain box.\n')
    #                     continue
    #             obj = patch_info['ann']
    #             if min(obj['bboxes'].shape) == 0:  # 张量为空
    #                 tmp_boxes = poly2obb(torch.tensor(obj['bboxes']), 'oc')  # oc转化可以处理空张量
    #             else:
    #                 tmp_boxes = poly2obb(torch.tensor(obj['bboxes']), self.version)  # 转化回5参数
    #             p_bboxes.append(tmp_boxes.to(device))
    #             # p_trunc.append(torch.tensor(obj['trunc'],device=device))  # 是否截断,box全部在win内部时为false
    #             ## 若box超出win范围则trunc为true
    #             p_labels.append(torch.tensor(patch_info['labels'], device=device))
    #             p_metas.append({'x_start': torch.tensor(patch_info['x_start'], device=device),
    #                             'y_start': torch.tensor(patch_info['y_start'], device=device),
    #                             'shape': patch_shape, 'trunc': torch.tensor(obj['trunc'], device=device),'img_shape': patch_shape, 'scale_factor': 1})

    #             patch = patchs[i]
    #             p_imgs.append(patch.to(device))

    #         out_imgs.append(p_imgs)
    #         out_bboxes.append(p_bboxes)
    #         out_labels.append(p_labels)
    #         out_metas.append(p_metas)

    #         #### change for sgdet
    #         # poly2obb(out_bboxes, self.version)
    #         return out_imgs, out_bboxes, out_labels, out_metas

    # elif mode == 'test':
    p_imgs = []
    p_metas = []
    img = imgs.cpu().squeeze(0)
    # patch
    info = dict()
    info['labels'] = np.array(torch.tensor([], device='cpu'))
    info['ann'] = {'bboxes': {}}
    info['width'] = img.shape[2]
    info['height'] = img.shape[1]

    sizes = [patch_shape[0]]
    # gaps=[0]
    windows = get_sliding_window(info, sizes, gaps, img_rate_thr)
    patchs, patch_infos = crop_img_withoutann(info, windows, img,
                                                no_padding=False,
                                                padding_value=padding_value)
    
    del img

    # # 对每张大图分解成的子图集合中的每张子图遍历
    # for i, patch_info in enumerate(patch_infos):
    #     p_metas.append({'x_start': torch.tensor(patch_info['x_start'], device=device),
    #                     'y_start': torch.tensor(patch_info['y_start'], device=device),
    #                     'shape': patch_shape, 'img_shape': patch_shape, 'scale_factor': 1})

    #     patch = patchs[i]
    #     p_imgs.append(patch)

    # out_imgs.append(p_imgs)
    # out_metas.append(p_metas)

    x_starts = torch.tensor([info['x_start'] for info in patch_infos], device=device)
    y_starts = torch.tensor([info['y_start'] for info in patch_infos], device=device)
    patch_shapes = [patch_shape for _ in patch_infos]  # Assuming patch_shape is a tensor
    img_shapes = [patch_shape for _ in patch_infos]  # Assuming patch_shape is a tensor
    scale_factors = [1 for _ in patch_infos]  # Assuming scale_factor is a tensor

    p_metas = [{'x_start': x, 'y_start': y, 'shape': s, 'img_shape': i, 'scale_factor': f} 
            for x, y, s, i, f in zip(x_starts, y_starts, patch_shapes, img_shapes, scale_factors)]

    p_imgs = [patch for patch in patchs]

    out_imgs.append(p_imgs)
    out_metas.append(p_metas)

    return out_imgs, out_metas

    # return out_imgs, out_bboxes, out_labels, out_metas

def list2tensor(img_lists):
    '''
    images: list of list of tensor images
    '''
    inputs = []
    for img in img_lists:
        inputs.append(img.cpu())
    inputs = torch.stack(inputs, dim=0).cpu()
    return inputs


def relocate(idx, local_bboxes, each_class, patch_meta):
    # 二阶段的bboxes为array
    # put patches' local bboxes to full img via patch_meta
    meta = patch_meta[idx]
    top = meta['y_start']
    left = meta['x_start']

    for in_class in range(len(local_bboxes)):
        local_bboxes_tmp = local_bboxes[in_class]
        each_class_tmp = each_class[in_class]
        
        if len(local_bboxes_tmp) != 0:
            local_bboxes_tmp[:, 0] += float(left)
            local_bboxes_tmp[:, 1] += float(top)
        
        if len(each_class_tmp) != 0: 
          each_class_tmp[:, :, 0] += float(left)
          each_class_tmp[:, :, 1] += float(top)          

        # for i in range(len(local_bboxes_tmp)):
        #     bbox = local_bboxes_tmp[i]
        #     # print('local_bboxes[i]:',bbox)
        #     bbox[0] += left
        #     bbox[1] += top
            
        #     each_box = each_class_tmp[i]
        #     each_box[:,0] = np.array((torch.tensor(each_box[:,0]).cuda() + left).cpu())
        #     each_box[:,1] = np.array((torch.tensor(each_box[:,1]).cuda() + top).cpu())

    return







# 从Global的信息整理成forward格式
def Collect_Global(g_img_infos, img_metas, length_thr):
    g_gt_boxes = []
    g_gt_labels = []

    for idx in range(len(g_img_infos)):
        g_gt_boxes.append(g_img_infos[idx]['gt_box'].squeeze(0))
        g_gt_labels.append(g_img_infos[idx]['labels'].squeeze(0))
        g_img_infos[idx]['img_shape'] = img_metas[0]['img_shape']
        g_img_infos[idx]['pad_shape'] = img_metas[0]['pad_shape']
        g_img_infos[idx]['scale_factor'] = 1.0

    # TODO:标签中会存在负值?
    # 各层按阈值进行标签分配(过滤)
    g_gt_boxes, g_gt_labels=filter_small_ann(g_gt_boxes, g_gt_labels, length_thr, g_img_infos)  # 这里进行标签过滤

    return g_gt_boxes, g_gt_labels, g_img_infos


def filter_small_ann(gt_bboxes, gt_labels, length_thr, g_img_infos=None):
    # 针对resize后图像中长度小于阈值的标签不保留
    # length_thr = 15

    gt_bboxes_global = []
    gt_labels_global = []
    gt_bboxes_global_ignore = []
    gt_labels_global_ignore = []
    # TODO:剔除resize后过小的标签,查看效果
    for gt, (bbox, label) in enumerate(zip(gt_bboxes, gt_labels)):
        # down_ratio = g_img_infos[gt]
        tmp_boxes = gt_bboxes[gt].clone()
        # gt_prepare = tmp_boxes[0].unsqueeze(0)  # 无gt时候补
        # gt_label_prepare = gt_labels[gt][[0]]
        gt_prepare = torch.zeros((0, 5), device=tmp_boxes.device)  # 无符合条件gt时来候补
        gt_label_prepare = torch.tensor([], device=tmp_boxes.device)
        # 根据长度阈值进行筛选
        mask = (tmp_boxes[:, 2] < length_thr) & (tmp_boxes[:, 3] < length_thr)

        tmp_boxes_out_ignore = tmp_boxes[mask]
        keeps_ignore = torch.nonzero(mask).squeeze(1)
        tmp_boxes_out = tmp_boxes[~mask]
        keeps = torch.nonzero(~mask).squeeze(1)

        tmp_labels_out = label[keeps]
        tmp_labels_out_ignore = label[keeps_ignore]

        if len(tmp_boxes_out) < 1:
            gt_bboxes_global.append(gt_prepare)
            gt_labels_global.append(gt_label_prepare)
        else:
            gt_bboxes_global.append(tmp_boxes_out)
            gt_labels_global.append(tmp_labels_out)

        gt_bboxes_global_ignore.append(tmp_boxes_out_ignore)
        gt_labels_global_ignore.append(tmp_labels_out_ignore)
    return gt_bboxes_global, gt_labels_global

@ROTATED_DETECTORS.register_module()
# class RotatedTwoStageDetector_Mul(BaseDetector):
class RotatedTwoStageDetector_Mul(nn.Module):
    """Base class for rotated two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone=None,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 ori_cfg= None):
        # super(RotatedTwoStageDetector_Mul, self).__init__(init_cfg,roi_head)
        super(RotatedTwoStageDetector_Mul, self).__init__()
        self.version = 'le90'
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained

        self.backbone = build_backbone(backbone)
        self.backbone_d2=build_backbone(backbone)
 
        ori_cfg  =  ori_cfg if  ori_cfg else init_cfg
        self.ori_cfg = ori_cfg if  ori_cfg else init_cfg

        if neck is not None:
            self.neck = build_neck(neck)
            self.neck_d2 = build_neck(neck)


        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)

            self.rpn_head = build_head(rpn_head_)
            self.rpn_head_d2 = build_head(rpn_head_)

        from maskrcnn_benchmark.modeling.roi_heads.roi_heads import build_roi_heads
        self.roi_heads = build_roi_heads(ori_cfg, ori_cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS)
        
        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained

            self.roi_head = build_head(roi_head)
            self.roi_head_d2 = build_head(roi_head)
        

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        ### discriminator tasks ###
        if self.ori_cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX and self.ori_cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.tasks = "Predcls"
        elif self.ori_cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX and not self.ori_cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.tasks = "Sgcls"
        elif not self.ori_cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX and not self.ori_cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.tasks = "Sgdets"
        

        self.all_extract_feat = [self.extract_feat, self.extract_feat_d2]
        self.all_RPN = [self.rpn_head, self.rpn_head_d2]
        self.all_ROI = [self.roi_head, self.roi_head_d2]

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img.cuda()) ### 1220
        # if self.with_neck:
        x = self.neck(x)
        return x

    def extract_feat_d2(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone_d2(img.cuda()) ### 1220
        # if self.with_neck:
        x = self.neck_d2(x)
        return x
    
    
    def Test_Patches_Img(self,img,patch_shape,gaps, p_bs, proposals, rescale=False):
        """
        对输入的img按patch_shape,gaps决定的窗口进行切块检测
        """
        # Crop full img into patches
        gt_bboxes=[]
        gt_labels=[]
        device=get_device()
        local_bboxes_lists=[]
        pathches_cls_scores = []
        l_all_box_cls = []
        p_imgs, p_metas = FullImageCrop(self, img, gt_bboxes, gt_labels,
                                        patch_shape=patch_shape,
                                        gaps=gaps, mode='test')
        temp = img.shape[0]
        H =  img.shape[2]
        W = img.shape[3]
        #if (img.shape[2] > 12000 or img.shape[3] > 12000):
        img = img.cpu()
        del img
        
        # print(len(p_metas[0]))
        for i in range(temp):
            j = 0
            patches = list2tensor(p_imgs[i])  # list to tensor
            patches_meta = p_metas[i]

            # patch batchsize
            while j < len(p_imgs[i]):
                if (j+p_bs) >= len(p_imgs[i]):
                    patch = patches[j:]
                    patch_meta = patches_meta[j:]
                else:
                    patch = patches[j:j + p_bs]
                    patch_meta = patches_meta[j:j + p_bs]  # x_start and y_start

                with torch.no_grad():
                    # fea_l_neck = self.extract_feat(patch)
                    patch=patch.to(device)
                    x = self.all_extract_feat[0](patch)

                    ##
                    patch = patch.cpu()
                    del patch

                    ###
                    if proposals is None:
                        proposal_list = self.all_RPN[0].simple_test_rpn(x, patch_meta)
                    else:
                        proposal_list = proposals
                    # 这里输出Local的预测结果
                    # outs_local = self.bbox_head(fea_l_neck)
                    # local的meta设置为1,因为这里未经过缩放
                    local_bbox_list, l_selec_cls_scores, new_all_box_cls = self.all_ROI[0].simple_test(
                        x, proposal_list, patch_meta, rescale=rescale,large = True)
                    
                    torch.cuda.empty_cache()
                    x = tuple(tensor.cpu() for tensor in x)
                    del x

                    # 将每个patch的local boxes放置到大图上对应的位置
                    for idx, (res_list,each_calss_local) in enumerate(zip(local_bbox_list,new_all_box_cls)):
                        det_bboxes = res_list
                        relocate(idx, det_bboxes, each_calss_local, patch_meta)
                    local_bboxes_lists.append(local_bbox_list)
                    pathches_cls_scores.append(l_selec_cls_scores)
                    l_all_box_cls.append(new_all_box_cls)
                    # local_bboxes_lists.append([local_bbox_list,local_label_list])

                j = j+p_bs

        #if (H > 12000 or W > 12000):

        bbox_list,p_keeps = merge_results_two_stage(local_bboxes_lists,iou_thr=0.4,flag=2)

        ################### 
        new_mer_cls_scores = hs(pathches_cls_scores,p_keeps)
        new_l_all_box_cls = hs(l_all_box_cls,p_keeps)        

        out_list = [tt if tt.shape[-1] == 6 else np.zeros((0, 6)) for tt in bbox_list]
                

        return out_list,new_mer_cls_scores,new_l_all_box_cls



    def Test_Concat_Patches_GlobalImg_without_fea(self, ori_img, ratio, scale, g_fea, patch_shape, gaps,
                                                   p_bs, proposals, gt_bboxes = None, gt_labels = None, rescale=False,id = None):
        """
        对按一定比例scale缩小后的global img进行切块检测,并返回拼接后的完整特征图
        Args:
            ratio: 当前金字塔某一层的global img大小和金字塔最顶层的img大小的比值
            scale: 原始图像的大小和当前金字塔某一层的global img大小的比值
        """
        device=get_device()
       
        if (ori_img.shape[2] > 8000 or ori_img.shape[3] > 8000):
            ori_img_cpu = ori_img.cpu()
            # print("move to cpu done!")
            img = F.interpolate(ori_img_cpu, scale_factor=1 / scale, mode='bilinear')
            img = img.to(device) 
            # print("move to gpu done!")
        else:
            img = F.interpolate(ori_img, scale_factor=1 / scale, mode='bilinear')
        
        # print('global img shpae:',img.shape)

        patches_bboxes_lists = []
        pathches_cls_scores = []
        g_all_box_cls = []
       
        gt_bboxes = []
        gt_labels = []
        p_imgs, p_metas = FullImageCrop(self, img, gt_bboxes, gt_labels, 
                                        patch_shape=patch_shape,gaps=gaps, mode='test')

        length = img.shape[0]
        torch.cuda.empty_cache()
        img = img.cpu()
        del img

        for i in range(length):
            j = 0
            patches = list2tensor(p_imgs[i])  # list to tensor
            patches_meta = p_metas[i]
            # patch batchsize
            while j < len(p_imgs[i]):
                if (j + p_bs) >= len(p_imgs[i]):
                    patch = patches[j:]
                    patch_meta = patches_meta[j:]
                else:
                    patch = patches[j:j + p_bs]
                    patch_meta = patches_meta[j:j + p_bs]  # x_start and y_start
              
                with torch.no_grad():
                    # fea_l_neck = self.extract_feat(patch)
                    patch=patch.to(device)
                    patch_fea = self.all_extract_feat[id](patch)

                    if proposals is None:
                        proposal_list = self.all_RPN[id].simple_test_rpn(patch_fea, patch_meta)
                    else:
                        proposal_list = proposals

                    # 这里输出每组patch上的预测结果
                    global_bbox_list, g_selec_cls_scores, new_all_box_cls = self.all_ROI[id].simple_test(
                        patch_fea, proposal_list, patch_meta, rescale=rescale,large = True)

                    patches_bboxes_lists.append(global_bbox_list)
                    pathches_cls_scores.append(g_selec_cls_scores)
                    g_all_box_cls.append(new_all_box_cls)
                    
                    
                    torch.cuda.empty_cache()
                    patch = patch.cpu()
                    patch_fea = tuple(tensor.cpu() for tensor in patch_fea)
                    del patch_fea, patch

                    for idx, (res_list,each_class) in enumerate(zip(global_bbox_list,new_all_box_cls)):
                        # 1)将每个patch的local boxes按照裁剪时的坐标放置到大图上对应的位置
                        relocate(idx, res_list,each_class, patch_meta)
                        # 2)按照缩放倍率放大到原始影像尺寸
                        resize_bboxes_len6(res_list,each_class,scale)

                        conf_thr=0.4
                        conf_thr=0
                        num_thr=0  # 数量阈值,global下检测到多于num_thr的则精细检测 
                        box_count=np.sum(res_list[0][:,-1]>conf_thr)
                        
                        if box_count>num_thr and scale==2: #只在down2阶段进行加速,更高层金字塔可能会不准确
                            # print('box_count:',box_count)
                            # if scale<8:  # 上采样倍数过高则误差率大
                            # print('可能含有实例的区域为:',patch_meta[idx])
                            # saved_p_metas.append(patch_meta[idx])
                            #TODO:使用global中检测到的大致位置来直接进行子图块的检测
                            p_meta=patch_meta[idx]
                            # print('p_meta:',p_meta)
                            left=int(p_meta['x_start']*scale) # 恢复到原图相对位置
                            top=int(p_meta['y_start']*scale)
                            shape=int(p_meta['shape'][0])

                            sub_meta_list=[]  # 存放4个子块的坐标
                            sub_img_list=[] # 存放4个子块
                            x0=left
                            x1=left+shape
                            y0=top
                            y1=top+shape
                            xy_list=[(x0,y0),(x0,y1),(x1,y0),(x1,y1)]

                            for xy in xy_list:
                                tmpx=xy[0]
                                tmpy=xy[1]
                                sub_meta=copy.deepcopy(p_meta)
                                sub_meta['x_start']=tmpx
                                sub_meta['y_start']=tmpy
                                sub_meta_list.append(sub_meta)
                                sub_img=ori_img[:,:,tmpy:tmpy+shape,tmpx:tmpx+shape].squeeze(0)
                                
                                if (sub_img.shape[-1] != 1024) or (sub_img.shape[-2] != 1024):
                                
                                    padd = np.empty((1024, 1024,sub_img.shape[0]),
                                                            dtype=np.float32)

                                    padd[...] =  [0.0081917211329, -0.004901960784, 0.0055655449953] 
                                    # 再交换到正确维度(3,800,800)
                                    padd=torch.tensor(padd.transpose((2,1,0)),
                                                            device=sub_img.device)
                                    
                                    padd[ ...,:sub_img.shape[1], :sub_img.shape[2]] = sub_img
                                    sub_img = padd
                
                                
                                
                                # sub_img=ori_img[:,:,tmpx:tmpx+shape,tmpy:tmpy+shape].squeeze(0)
                                sub_img_list.append(sub_img)
                            
                            # print('sub_meta_list:',sub_meta_list)
                                
                            sub_img=list2tensor(sub_img_list)
                            sub_img=sub_img.to(device)

                            sub_x = self.all_extract_feat[0](sub_img)

                            if proposals is None:
                                    proposal_list = self.all_RPN[0].simple_test_rpn(sub_x, sub_meta_list)
                            else:
                                proposal_list = proposals
                            
                            sub_bbox_list,sub_selec_cls_scores, sub_all_box_cls = self.all_ROI[0].simple_test(
                                    sub_x, proposal_list, sub_meta_list, rescale=rescale,large = True)



                    
                            torch.cuda.empty_cache()
                            sub_img = sub_img.cpu()
                            sub_x = tuple(tensor.cpu() for tensor in sub_x)
                            del sub_x, sub_img

                            # # 将每个patch的local boxes放置到大图上对应的位置
                            for idx, (res_list,sub_class) in enumerate(zip(sub_bbox_list,sub_all_box_cls)):
                               
                              
                                relocate(idx, res_list,sub_class, sub_meta_list)
                            
                            # 添加底层预测结果
                            patches_bboxes_lists.append(sub_bbox_list)
                            pathches_cls_scores.append(sub_selec_cls_scores)
                            g_all_box_cls.append(sub_all_box_cls)

                
                j = j + p_bs
        
        

        del p_metas

        patches_bboxes_list,p_keeps = merge_results_two_stage(patches_bboxes_lists, iou_thr=0.4,flag=1)
        
        new_mer_cls_scores = hs(pathches_cls_scores,p_keeps)
        new_g_all_box_cls = hs(g_all_box_cls,p_keeps)

        out_list = [tt if tt.shape[-1] == 6 else np.zeros((0, 6)) for tt in patches_bboxes_list]

        full_patches_out =[]

        return out_list, full_patches_out, new_mer_cls_scores,new_g_all_box_cls



    def simple_test(self, img, img_metas,gt_bboxes= None, gt_labels =None, proposals=None, rescale=False):
        """Test without augmentation."""


        global_shape_h = img.shape[2]
        global_shape_w = img.shape[3]
        
        if global_shape_h > 10000 or global_shape_w > 10000:
            p_bs  = 1
            p_bs_2 = 1
        else:
            p_bs = 2
            p_bs_2 = 4

        gaps = [200]
        patch_shape = (1024, 1024)

        global_shape_max=max(global_shape_h,global_shape_w)
        
        if global_shape_max <= 1024:  # all

            local_bboxes_list,local_each_cls_scores,l_box_en = self.Test_Patches_Img(img, patch_shape, gaps, p_bs_2, proposals, rescale=False)

            local_bboxes = [local_bboxes_list] #torch.cat(filter_local_bboxes_list)

            all =  local_bboxes
            #### 合并
            all_scores =   [local_each_cls_scores]
            all_en =  [l_box_en]

            all_nms, all_keeps = merge_results_two_stage(all, iou_thr=0.4,flag=3)
            new_mer_cls_scores = hs_all(all_scores,all_keeps)
            new_en = hs_all(all_en,all_keeps)               
            
      

        else:

            global_shape_list = []   ## for speed
            while global_shape_max > 1024:
                global_shape_h >>= 1
                global_shape_w >>= 1
                global_shape_max >>= 1
                global_shape_list.append((global_shape_h, global_shape_w))

                
            global_shape_min = (global_shape_h,global_shape_w)
            

            # 2)再依次向下层得到切块结果以及对应的整特征图
            all_bboxes_lists=[]
            gaps = [200]
            patch_shape = (1024, 1024)
        
            # p_bs = 4  # patch batchsize
            global_fea_list = []
            global_each_cls_scores = []
            g_box_en = []

            level_num = 0

            for global_shape in global_shape_list:
                # scale: 原始大幅面图像img的大小和当前金字塔某一层的global img大小的比值
                scale = img.shape[3]/global_shape[1]
                # ratio: 当前金字塔某一层的global img大小和金字塔最顶层的global img大小的比值
                ratio = global_shape[0]/global_shape_min[0]
                
                # TODO: 控制预测的下采样层数
                scale_int=int(scale)
                

                # TODO:注意这里注释掉了Test_Concat_Patches_GlobalImg中的拼合特征的代码,以便于测试
                global_patches_bbox_list, global_full_fea, each_cls_scores,each_box_en = self.Test_Concat_Patches_GlobalImg_without_fea(img, ratio, scale,
                                                                                            None,
                                                                                            patch_shape, gaps, p_bs,
                                                                                            proposals,gt_bboxes=gt_bboxes,gt_labels=gt_labels,id = level_num)
                all_bboxes_lists.append(global_patches_bbox_list)
                global_fea_list.append(global_full_fea)
                global_each_cls_scores.append(each_cls_scores)
                g_box_en.append(each_box_en)

                level_num = 1


            
            all = all_bboxes_lists

            #### 合并
            all_scores =  global_each_cls_scores 
            all_en = g_box_en

            all_nms, all_keeps = merge_results_two_stage(all, iou_thr=0.4,flag=3)
            new_mer_cls_scores = hs_all(all_scores,all_keeps)
            new_en = hs_all(all_en,all_keeps)               
            
            
                
        all_nms_list = [tt if tt.shape[-1] == 6 else np.zeros((0, 6)) for tt in all_nms]
        
        return [all_nms_list] ,new_mer_cls_scores,new_en
    


    def batch(self,img,targets):
            imgs = [img]
            if  not self.training:
                img_metas = [[targets[0].extra_fields["data"]["img_metas"][0].data]]
            else:
                 img_metas = [[targets[0].extra_fields["data"]["img_metas"].data]]
    
            for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
                if not isinstance(var, list):
                    raise TypeError(f'{name} must be a list, but got {type(var)}')

            num_augs = len(imgs)
            if num_augs != len(img_metas):
                raise ValueError(f'num of augmentations ({len(imgs)}) '
                                f'!= num of image meta ({len(img_metas)})')
        
            # NOTE the batched image size information may be useful, e.g.
            # in DETR, this is needed for the construction of masks, which is
            # then used for the transformer_head.
            for img, img_meta in zip(imgs, img_metas):
                batch_size = len(img_meta)
                for img_id in range(batch_size):
                    
                    img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])


            if num_augs == 1:
        
                results, cls_scores,new_en = self.simple_test(imgs[0], img_metas[0])


            # return results
            ### 删选 按照得分>0.3
            ############################## 0.3
            sclec_id = []
            f_results = []
            f_cls_scores = []
            f_en = []

            for k1 in range(len(results[0])):
                sclec_id.append([])
                # f_results.append([])
                # f_cls_scores.append([])
                get_data = results[0][k1]
                cls1 = cls_scores[k1]
                conf = get_data[:,5]
                en = new_en[k1]
                pos =  np.where(conf >= 0.3)[0]
                if len(pos) == 0:
                    sclec_id[k1].append(-500)
                    f_results.append([])
                    f_cls_scores.append([])
                    f_en.append([])
                    continue
                else:
                    f_en.append(en[pos])
                    f_results.append(get_data[pos])
                    f_cls_scores.append(cls1[pos])
                    sclec_id[k1].append(pos)
            ############################## 0.3
                    
            ##  合并所有bbox
            no_f_results = [f for f in f_results if len(f) != 0]

            if len(no_f_results) == 0 or (len(no_f_results)==1 and len(no_f_results[0] == 1)):
                ############################## 0.2
                sclec_id = []
                f_results = []
                f_cls_scores = []
                f_en = []
                for k1 in range(len(results[0])):
                    sclec_id.append([])
                    # f_results.append([])
                    # f_cls_scores.append([])
                    get_data = results[0][k1]
                    cls1 = cls_scores[k1]
                    conf = get_data[:,5]
                    en = new_en[k1]
                    pos =  np.where(conf >= 0.2)[0]
                    if len(pos) == 0:
                        sclec_id[k1].append(-500)
                        f_results.append([])
                        f_cls_scores.append([])
                        f_en.append([])
                        continue
                    else:
                        f_en.append(en[pos])
                        f_results.append(get_data[pos])
                        f_cls_scores.append(cls1[pos])
                        sclec_id[k1].append(pos)
                ############################## 0.2
                no_f_results = [f for f in f_results if len(f) != 0]
                if len(no_f_results) == 0 or (len(no_f_results)==1 and len(no_f_results[0] == 1)):
                    sclec_id = []
                    f_results = []
                    f_cls_scores = []
                    f_en = []
                    for k1 in range(len(results[0])):
                        sclec_id.append([])
                        # f_results.append([])
                        # f_cls_scores.append([])
                        get_data = results[0][k1]
                        cls1 = cls_scores[k1]
                        conf = get_data[:,5]
                        en = new_en[k1]
                        pos =  np.where(conf >= 0.1)[0]
                        if len(pos) == 0:
                            sclec_id[k1].append(-500)
                            f_results.append([])
                            f_cls_scores.append([])
                            f_en.append([])
                            continue
                        else:
                            f_en.append(en[pos])
                            f_results.append(get_data[pos])
                            f_cls_scores.append(cls1[pos])
                            sclec_id[k1].append(pos)
                    no_f_results = [f for f in f_results if len(f) != 0]
                    
                    if len(no_f_results) == 0 or (len(no_f_results)==1 and len(no_f_results[0] == 1)):
                        sclec_id = []
                        f_results = []
                        f_cls_scores = []
                        f_en = []
                        for k1 in range(len(results[0])):
                            sclec_id.append([])
                            # f_results.append([])
                            # f_cls_scores.append([])
                            get_data = results[0][k1]
                            cls1 = cls_scores[k1]
                            conf = get_data[:,5]
                            en = new_en[k1]
                            pos =  np.where(conf >= 0.001)[0]
                            if len(pos) == 0:
                                sclec_id[k1].append(-500)
                                f_results.append([])
                                f_cls_scores.append([])
                                f_en.append([])
                                continue
                            else:
                                f_en.append(en[pos])
                                f_results.append(get_data[pos])
                                f_cls_scores.append(cls1[pos])
                                sclec_id[k1].append(pos)
                        no_f_results = [f for f in f_results if len(f) != 0]

                        if len(no_f_results) == 0 or (len(no_f_results)==1 and len(no_f_results[0] == 1)):
                            sclec_id = []
                            f_results = []
                            f_cls_scores = []
                            f_en = []
                            for k1 in range(len(results[0])):
                                sclec_id.append([])
                                # f_results.append([])
                                # f_cls_scores.append([])
                                get_data = results[0][k1]
                                cls1 = cls_scores[k1]
                                conf = get_data[:,5]
                                en = new_en[k1]
                                pos =  np.where(conf >= 0.00001)[0]
                                if len(pos) == 0:
                                    sclec_id[k1].append(-500)
                                    f_results.append([])
                                    f_cls_scores.append([])
                                    f_en.append([])
                                    continue
                                else:
                                    f_en.append(en[pos])
                                    f_results.append(get_data[pos])
                                    f_cls_scores.append(cls1[pos])
                                    sclec_id[k1].append(pos)
                            no_f_results = [f for f in f_results if len(f) != 0]



            no_cls_score = [f1 for f1 in f_cls_scores if len(f1) != 0]
            np_en = [f2 for f2 in f_en if len(f2) != 0]

            if len(no_f_results) == 0:
                return 666,None
            else:
                all_box = torch.tensor(np.concatenate(no_f_results,axis=0)).cuda()

            all_score = torch.tensor(np.concatenate(no_cls_score,axis=0)).cuda()
            all_en = torch.tensor(np.concatenate(np_en,axis=0)).cuda()

            all_score = all_score[:,POS]

            proposals = copy.deepcopy(targets[0])
            if 0 in all_box[:,0]:
                zero_list = np.where(all_box[:,0].cpu().numpy()==0)[0].tolist()
                N = list(range(len(all_box)))
                N_s = [x for x in N if x not in zero_list]  
                all_box = all_box[N_s]
                all_score = all_score[N_s]
                all_en = all_en[N_s]
            elif 0 in all_box[:,1]:
                zero_list = np.where(all_box[:,1].cpu().numpy()==0)[0].tolist()
                N = list(range(len(all_box)))
                N_s = [x for x in N if x not in zero_list]  
                all_box = all_box[N_s]
                all_score = all_score[N_s]
                all_en = all_en[N_s]
            elif 0 in all_box[:,2]:
                zero_list = np.where(all_box[:,2].cpu().numpy()==0)[0].tolist()
                N = list(range(len(all_box)))
                N_s = [x for x in N if x not in zero_list]  
                all_box = all_box[N_s]
                all_score = all_score[N_s]
                all_en = all_en[N_s]
            elif 0 in all_box[:,3]:
                zero_list = np.where(all_box[:,3].cpu().numpy()==0)[0].tolist()
                N = list(range(len(all_box)))
                N_s = [x for x in N if x not in zero_list]  
                all_box = all_box[N_s]
                all_score = all_score[N_s]
                all_en = all_en[N_s]


            ####
            proposals.bbox = all_box[:,:5]


            proposals.extra_fields["predict_logits"] = all_score
            proposals.extra_fields["boxes_per_cls"] = all_en
            assert len(all_en) == len(all_score)
            del proposals.extra_fields["labels"]
            # del proposals.extra_fields["attributes"]
            del proposals.extra_fields["data1"]
            del proposals.extra_fields["target1"]
            del proposals.extra_fields["relation"]



            if self.ori_cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "RPCM" or "HetSGG_Predictor":
                logits = all_score
                
                # 使用 PyTorch 的 argmax 函数以避免将数据移动回 CPU
                all_labels = torch.argmax(logits[:, 1:], dim=1) + 1
                
                proposals.extra_fields["pred_labels"] = all_labels

                obj_scores = torch.softmax(logits, 1).detach()
                obj_score_ind = torch.arange(logits.shape[0], device=obj_scores.device) * logits.shape[
                    1] + all_labels
                obj_scores = obj_scores.view(-1)[obj_score_ind]
                proposals.add_field("pred_scores", obj_scores)



            ### 分配标签
            # if self.training:
            # if ite == 19:
            #     t = 1
            iou = torch.tensor(box_iou_rotated(  # 533,1003
                targets[0].bbox.float(),
                proposals.bbox.float()).cpu().numpy()).cuda()
            matched_vals, matches = iou.max(dim=0)
            below_low_threshold = matched_vals < 0.3 #self.low_threshold
            between_thresholds = (matched_vals >= 0.3) & (
                matched_vals <  0.5 #self.high_threshold
            )
            matches[below_low_threshold] = -1 #  Matcher.BELOW_LOW_THRESHOLD
            matches[between_thresholds] = -2 #Matcher.BETWEEN_THRESHOLD
            matched_idxs = matches
            target = targets[0].copy_with_fields(["labels"])
            matched_targets = target[matched_idxs.clamp(min=0)]
            labels_per_image = matched_targets.get_field("labels").to(dtype=torch.int64)
            # attris_per_image = matched_targets.get_field("attributes").to(dtype=torch.int64)
            labels_per_image[matched_idxs < 0] = 0
            # attris_per_image[matched_idxs < 0, :] = 0
            proposals.add_field("labels", labels_per_image) #all_labels
            #proposals.add_field("labels", all_labels)
            # proposals.add_field("attributes", attris_per_image)
            
            # w, h = targets[0].size[0], targets[0].size[1]
            # w_f, h_f = 1024/w, 1024/h
            b1 = targets[0].bbox


            b2 = targets[0].extra_fields["target1"].bbox
            w_f = float((b2[:,0] / b1[:,0])[0])
            h_f = float((b2[:,1] / b1[:,1])[0])
            hw_f = float((b2[:,2] / b1[:,2])[0])
      
            if self.training:
                s_size = targets[0].extra_fields["data1"]["img_metas"].data["pad_shape"]
            else:
                s_size = targets[0].extra_fields["data1"]["img_metas"][0].data["pad_shape"]
            sh,sw =  s_size[0],s_size[1]
            
            proposals.bbox[:,0] *= w_f
            proposals.bbox[:,1] *= h_f
            proposals.bbox[:,2:4] *= hw_f
            proposals.size = (sw,sh)

            if not self.training:
                return proposals,[w_f,h_f,hw_f]
            return proposals,None



    def forward(self, img, targets=None, logger=None, ite=None,  gt_bboxes_ignore=None, gt_masks=None, proposals=None, sgd_data = None, MUL = None,m = None,val = None,
                vae = None, bce = None, **kwargs): 
 
        imgs = img.tensors

        if self.tasks == "Predcls":  
            losses = dict()
            x = self.extract_feat(imgs) # feature
            proposals = targets
            if self.roi_heads:  ### relation
                 if self.ori_cfg.CFA_pre == 'extract_aug':
                    tail_dict = self.roi_heads(x, proposals, targets, logger,ite=ite,OBj = self.roi_head)
                    return tail_dict
                 x, result, detector_losses = self.roi_heads(x, proposals, targets, logger,
                                                        ite=ite,OBj = self.roi_head,MUL =MUL,m=m,val = val,vae = vae,bce = bce)
            if self.training:     
                losses.update(detector_losses)
                return losses 
            else:
                return result
        
        elif self.tasks == "Sgcls" : #or self.tasks =="Sgdets": ### need to predict class labels
            if len(targets) != 1:
                img_metas = []
                gt_bboxes = []
                gt_labels = []
                for tar in targets:
                     img_metas.append(tar.extra_fields["data"]["img_metas"].data if self.training else tar.extra_fields["data"]["img_metas"][0].data)
                     gt_bboxes.append(tar.extra_fields["data"]["gt_bboxes"].data.cuda() if self.training else tar.extra_fields["data"]["gt_bboxes"][0].data.cuda())
                     gt_labels.append(tar.extra_fields["data"]["gt_labels"].data.long().cuda() if self.training else tar.extra_fields["data"]["gt_labels"][0].data.long().cuda())  
            else:
                img_metas =  [ targets[0].extra_fields["data"]["img_metas"].data] if self.training else [ targets[0].extra_fields["data"]["img_metas"][0].data]
                if self.ori_cfg.CFA_pre == 'extract_aug':
                    gt_bboxes =  [ targets[0].extra_fields["data"]["gt_bboxes"].data.cuda() ]
                else:
                    gt_bboxes = [ targets[0].extra_fields["data"]["gt_bboxes"].data.cuda()] if self.training else [ targets[0].extra_fields["data"]["gt_bboxes"][0].data.cuda() ]
                gt_labels = [ targets[0].extra_fields["data"]["gt_labels"].data.long().cuda()] if self.training else [ targets[0].extra_fields["data"]["gt_labels"][0].data.cuda() ]
            losses = dict()
            x = self.extract_feat(imgs) # feature
            proposals = targets


            
            bbox_results = self.roi_head.forward_train(x, img_metas, proposals,
                                                    gt_bboxes, gt_labels,
                                                    gt_bboxes_ignore, gt_masks, flag = True,
                                                         **kwargs)
            
            cls_score = bbox_results["cls_score"] 
            cls_score = cls_score[:,POS]
            
            start = 0    
            for pro in proposals:   
                lens = len(pro)
                pro.extra_fields["predict_logits"] = cls_score[start : start+lens,:]
                start = lens

            if self.ori_cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "GCN_RELATION" or "HetSGG_Predictor":

                for pro in proposals: 
                    # 将 extra_fields["predict_logits"] 转换为一个 PyTorch 张量
                    logits = pro.extra_fields["predict_logits"]
                    
                    # 使用 PyTorch 的 argmax 函数以避免将数据移动回 CPU
                    all_labels = torch.argmax(logits[:, 1:], dim=1) + 1
                    
                    pro.extra_fields["pred_labels"] = all_labels

                    obj_scores = torch.softmax(logits, 1).detach()
                    obj_score_ind = torch.arange(logits.shape[0], device=obj_scores.device) * logits.shape[
                        1] + all_labels
                    obj_scores = obj_scores.view(-1)[obj_score_ind]
                    pro.add_field("pred_scores", obj_scores)

            if self.roi_heads:  ### relation
                 x, result, detector_losses = self.roi_heads(x, proposals, targets, logger,
                                                        ite=ite,OBj = self.roi_head,MUL = MUL,m=m,val = val,vae = vae)

            if self.training:     
                losses.update(detector_losses)
                return losses 
            else:
                return result
            
        elif self.tasks == "Sgdets":
     
            p = []
            sf_img = []
            sf_tar = []
            s_f = []
            
            for k in range(len(imgs)):
                det_box, scale= self.batch(imgs[k].unsqueeze(0),[targets[k]])
                p.append(det_box)
                s_f.append(scale)
                
            losses = dict()
           
            x = self.extract_feat(sgd_data[0].tensors) # feature
            ##
            if self.roi_heads:  ### relation
                 x, result, detector_losses = self.roi_heads(x, p, sgd_data[1], logger,
                                                        ite=ite,OBj = self.roi_head,s_f = s_f,MUL =MUL,m=m,val = val,vae = vae)

            if self.training:     
                losses.update(detector_losses)
                return losses 
            else:
                return result
            
    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
