# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector

from mmdet.utils import get_device
from .img_split_bridge_tools_hbb import *
from mmdet.core.bbox.transforms import (hbb2obb,obb2hbb,obb2poly_oc,poly2obb_oc)
import numpy as np
import torch.nn.functional as F


# from sahi.model import MmdetDetectionModel
# mmdet_faster_rcnn_model_path = 'epoch_18.pth'
# mmdet_faster_rcnn_config_path = 'mmdetection20/configs/faster_rcnn/faster_rcnn_r50_fpn_2x_allbridge.py'
# detection_model = MmdetDetectionModel(
# model_path=mmdet_faster_rcnn_model_path,
# config_path=mmdet_faster_rcnn_config_path,
# device="cuda")

def resize_bboxes_len5(bboxes_out,scale):
    """Resize bounding boxes with scales."""

    for i in range(len(bboxes_out)):
        box_out=bboxes_out[i]
        w_scale = scale
        h_scale = scale
        box_out[:, 0] *= w_scale
        box_out[:, 1] *= h_scale
        box_out[:, 2] *= w_scale
        box_out[:, 3] *= h_scale
        # box_out[:, 2:4] *= np.sqrt(w_scale * h_scale)

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
    img_rate_thr = 0.6   
    iof_thr = 0.1  
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
            info['ann']['bboxes'] = np.array(hbb2obb(tmp_boxes, 'oc'))         
            info['ann']['bboxes'] = np.array(obb2poly_oc(tmp_boxes))  
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

                      
            for i, patch_info in enumerate(patch_infos):
                if jump_empty_patch:
                    
                    if patch_info['labels'] == [-1]:
                        # print('Patch does not contain box.\n')
                        continue
                obj = patch_info['ann']
                if min(obj['bboxes'].shape) == 0:  
                    tmp_boxes = poly2obb_oc(torch.tensor(obj['bboxes']))                     
                    tmp_boxes=obb2hbb(tmp_boxes,'oc')
                else:
                    tmp_boxes = poly2obb_oc(torch.tensor(obj['bboxes']))  
                    tmp_boxes = obb2hbb(tmp_boxes, 'oc')
                p_bboxes.append(tmp_boxes.to(device))
                # p_trunc.append(torch.tensor(obj['trunc'],device=device))  
               
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

       
        for i, patch_info in enumerate(patch_infos):
            p_metas.append({'x_start': torch.tensor(patch_info['x_start'], device=device),
                            'y_start': torch.tensor(patch_info['y_start'], device=device),
                            'ori_shape': patch_shape,
                            'shape': patch_shape, 'img_shape': patch_shape, 'scale_factor': 1})

            patch = patchs[i]
            p_imgs.append(patch.cpu())

        out_imgs.append(p_imgs)
        out_metas.append(p_metas)

        return out_imgs, out_metas

    return out_imgs, out_bboxes, out_labels, out_metas

def list2tensor(img_lists):
    '''
    images: list of list of tensor images
    '''
    inputs = []
    for img in img_lists:
        inputs.append(img.cpu())
    inputs = torch.stack(inputs, dim=0)
    return inputs

def relocate(idx, local_bboxes, patch_meta):
    
    # local_boxes:(n,5):(x0,y0,x1,y1,score)
    # put patches' local bboxes to full img via patch_meta
    meta = patch_meta[idx]
    top = meta['y_start'].cpu().numpy()
    left = meta['x_start'].cpu().numpy()

    local_bboxes_tmp = local_bboxes[0]
    local_bboxes_tmp[:, 0] += left
    local_bboxes_tmp[:, 1] += top
    local_bboxes_tmp[:, 2] += left
    local_bboxes_tmp[:, 3] += top
    return

def filter_small_ann(gt_bboxes, gt_labels, length_thr, g_img_infos=None):
    # length_thr = 15

    gt_bboxes_global = []
    gt_labels_global = []
    gt_bboxes_global_ignore = []
    gt_labels_global_ignore = []
    
    for gt, (bbox, label) in enumerate(zip(gt_bboxes, gt_labels)):
        # down_ratio = g_img_infos[gt]
        tmp_boxes = gt_bboxes[gt].clone()
        # gt_prepare = tmp_boxes[0].unsqueeze(0)  
        # gt_label_prepare = gt_labels[gt][[0]]
        gt_prepare = torch.zeros((0, 4), device=tmp_boxes.device)  
        gt_label_prepare = torch.tensor([], device=tmp_boxes.device)
               
        mask = ((tmp_boxes[:, 2]-tmp_boxes[:, 0]) < length_thr) & ((tmp_boxes[:, 3]-tmp_boxes[:, 1]) < length_thr)

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

@DETECTORS.register_module()
class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TwoStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        self.global_backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)
            self.global_neck = build_neck(neck)


        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)
            self.global_rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)
            self.global_roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x
    
    def extract_feat_global(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.global_backbone(img)
        if self.with_neck:
            x = self.global_neck(x)
        return x
    
    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    # def forward_train(self,
    #                   img,
    #                   img_metas,
    #                   gt_bboxes,
    #                   gt_labels,
    #                   gt_bboxes_ignore=None,
    #                   gt_masks=None,
    #                   proposals=None,
    #                   **kwargs):
    #     x = self.extract_feat(img)
    #     losses = dict()

    #     # RPN forward and loss
    #     if self.with_rpn:
    #         proposal_cfg = self.train_cfg.get('rpn_proposal',
    #                                           self.test_cfg.rpn)
    #         rpn_losses, proposal_list = self.rpn_head.forward_train(
    #             x,
    #             img_metas,
    #             gt_bboxes,
    #             gt_labels=None,
    #             gt_bboxes_ignore=gt_bboxes_ignore,
    #             proposal_cfg=proposal_cfg,
    #             **kwargs)
    #         losses.update(rpn_losses)
    #     else:
    #         proposal_list = proposals

    #     roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
    #                                              gt_bboxes, gt_labels,
    #                                              gt_bboxes_ignore, gt_masks,
    #                                              **kwargs)
    #     losses.update(roi_losses)

    #     return losses
    
    # Train global detector
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        length_thr=15
        g_gt_boxes, g_gt_labels=filter_small_ann(gt_bboxes, gt_labels,length_thr)
        x = self.extract_feat_global(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.global_rpn_head.forward_train(
                x,
                img_metas,
                g_gt_boxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.global_roi_head.forward_train(x, img_metas, proposal_list,
                                                 g_gt_boxes,g_gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)
        # x = self.extract_feat(img)

        # losses = dict()

        # # RPN forward and loss
        # if self.with_rpn:
        #     proposal_cfg = self.train_cfg.get('rpn_proposal',
        #                                       self.test_cfg.rpn)
        #     rpn_losses, proposal_list = self.rpn_head.forward_train(
        #         x,
        #         img_metas,
        #         gt_bboxes,
        #         gt_labels=None,
        #         gt_bboxes_ignore=gt_bboxes_ignore,
        #         proposal_cfg=proposal_cfg,
        #         **kwargs)
        #     losses.update(rpn_losses)
        # else:
        #     proposal_list = proposals

        # roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
        #                                          gt_bboxes, gt_labels,
        #                                          gt_bboxes_ignore, gt_masks,
        #                                          **kwargs)
        # losses.update(roi_losses)
        
        # for debug         
        t = self.simple_test(self, img, img_metas)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def Test_Patches_Img(self,img,patch_shape,gaps, p_bs, proposals, rescale=False):
        # Crop full img into patches
        gt_bboxes=[]
        gt_labels=[]
        local_bboxes_lists=[]
        p_imgs, p_metas = FullImageCrop(self, img, gt_bboxes, gt_labels,
                                        patch_shape=patch_shape,
                                        gaps=gaps, mode='test')

        for i in range(img.shape[0]):
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
                    patch=patch.cuda()
                    x = self.extract_feat(patch)
                    if proposals is None:
                        proposal_list = self.rpn_head.simple_test_rpn(x, patch_meta)
                    else:
                        proposal_list = proposals
                                      
                    # outs_local = self.bbox_head(fea_l_neck)
                                     
                    local_bbox_list = self.roi_head.simple_test(
                        x, proposal_list, patch_meta, rescale=rescale)
                                      
                    for idx, res_list in enumerate(local_bbox_list):
                        det_bboxes = res_list
                        relocate(idx, det_bboxes, patch_meta)
                    local_bboxes_lists.append(local_bbox_list)
                    # local_bboxes_lists.append([local_bbox_list,local_label_list])

                j = j+p_bs
       
        bbox_list = merge_results_two_stage_hbb(local_bboxes_lists,iou_thr=0.4)
        
        print('local_patches_bboxes_list shape:',bbox_list.shape)
        if bbox_list.shape[-1]!=5:
            bbox_list = torch.zeros((0, 5)).cpu()

        return bbox_list

    def Test_Concat_Patches_GlobalImg(self, ori_img, ratio, scale, g_fea, patch_shape, gaps, p_bs, proposals, rescale=False):

        img = F.interpolate(ori_img, scale_factor=1 / scale, mode='bilinear')
        print('global img shpae:',img.shape)
        patches_bboxes_lists = []
        gt_bboxes = []
        gt_labels = []
        device=get_device()
        p_imgs, p_metas = FullImageCrop(self, img, gt_bboxes, gt_labels, 
                                        patch_shape=patch_shape,gaps=gaps, mode='test')

        for i in range(img.shape[0]):
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
                    patch_fea = self.extract_feat_global(patch)

                    if proposals is None:
                        proposal_list = self.global_rpn_head.simple_test_rpn(patch_fea, patch_meta)
                    else:
                        proposal_list = proposals
                    
                    global_bbox_list = self.global_roi_head.simple_test(
                        patch_fea, proposal_list, patch_meta, rescale=rescale)

                    for idx, res_list in enumerate(global_bbox_list):
                                           
                        relocate(idx, res_list, patch_meta)
                                             
                        resize_bboxes_len5(res_list, scale)

                    patches_bboxes_lists.append(global_bbox_list)
                j = j + p_bs

        patches_bboxes_list = merge_results_two_stage_hbb(patches_bboxes_lists, iou_thr=0.4)
        # print('scale:',scale)
        print('global_patches_bboxes_list shape:',patches_bboxes_list.shape)
        if patches_bboxes_list.shape[-1]!=5:
            patches_bboxes_list = torch.zeros((0, 5), device=device)
        # full_patches_out = [full_patch.cpu() for full_patch in full_patches]  #
        full_patches_out =[]
        return patches_bboxes_list, full_patches_out

    # def simple_test(self, img, img_metas, proposals=None, rescale=False):
    #     """Test without augmentation."""
    #     # 
    #     from sahi .predict import get_sliced_prediction
    #     from sahi import AutoDetectionModel
    #     from sahi.predict import get_prediction, get_sliced_prediction, predict
    #     from IPython.display import Image

    #     assert self.with_bbox, 'Bbox head must be implemented.'

    #     # crop patch
    #     # # (   
    # gaps = [200]
    #     # patch_shape = (1024, 1024)
    #     # p_bs = 4  # patch batchsize
    #     # local_bboxes_list = self.Test_Patches_Img(img, patch_shape, gaps, p_bs, proposals, rescale=False)
    #     # final_bbox_list=[local_bboxes_list.numpy()]

    #     # add sahi
    #     # mmdet_faster_rcnn_model_path = 'epoch_18.pth'
    #     # mmdet_faster_rcnn_config_path = 'mmdetection20/configs/faster_rcnn/faster_rcnn_r50_fpn_2x_allbridge.py'
    #     # detection_model = MmdetDetectionModel(
    #     # model_path=mmdet_faster_rcnn_model_path,
    #     # config_path=mmdet_faster_rcnn_config_path,
    #     # device="cuda")
    #     result = get_sliced_prediction(
    #         img_metas[0]['img_path'],
    #         detection_model,
    #         perform_standard_pred = True,
    #         slice_height = 1024,
    #         slice_width = 1024,
    #         overlap_height_ratio = 0.1953125,
    #         overlap_width_ratio = 0.1953125,
    #         postprocess_type = "NMS",
    #         postprocess_match_threshold=0.4 
    #     )
    #     object_prediction_list = result.object_prediction_list
    #     bbox_and_score = np.array([
    #     [
    #         prediction.bbox.minx,
    #         prediction.bbox.miny,
    #         prediction.bbox.maxx,
    #         prediction.bbox.maxy,
    #         prediction.score.value
    #     ]
    #     for prediction in object_prediction_list])

    #     print('bbox_and_score.shape:',bbox_and_score.shape)

    #     return [bbox_and_score]

    # Test local
    # def simple_test(self, img, img_metas, proposals=None, rescale=False):
    #     """Test without augmentation."""

    #     assert self.with_bbox, 'Bbox head must be implemented.'
    #     x = self.extract_feat(img)
    #     if proposals is None:
    #         proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
    #     else:
    #         proposal_list = proposals

    #     return self.roi_head.simple_test(
    #         x, proposal_list, img_metas, rescale=rescale)
    
    # Test global
    # def simple_test(self, img, img_metas, proposals=None, rescale=False):
    #     """Test without augmentation."""

    #     assert self.with_bbox, 'Bbox head must be implemented.'
    #     x = self.extract_feat_global(img)
    #     if proposals is None:
    #         proposal_list = self.global_rpn_head.simple_test_rpn(x, img_metas)
    #     else:
    #         proposal_list = proposals

    #     return self.global_roi_head.simple_test(
    #         x, proposal_list, img_metas, rescale=rescale)
    


# #
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        #import pdb;pdb.set_trace()
        assert self.with_bbox, 'Bbox head must be implemented.'
        all_bboxes_lists=[]

        # 
        # 
        global_shape_min = img.shape[3]
        
        gloabl_shape_list=[]
        # if int(img.shape[3])<10000:
        #     gloabl_shape_list.append((global_shape_min*1.2, global_shape_min*1.2))
        #     print('11')
        count=0
        while global_shape_min > 1024:
            global_shape_min = global_shape_min/2
            gloabl_shape_list.append((global_shape_min, global_shape_min))
            count+=1
        #     if count>2:
        #         break
       
        global_shape_min = (global_shape_min, global_shape_min)
        print('global_shape_min',global_shape_min )
        scale_min = img.shape[3] / global_shape_min[0]
        global_img_min = F.interpolate(img, scale_factor=1 / scale_min, mode='bilinear')

        min_g_feature = self.extract_feat(global_img_min)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(min_g_feature, img_metas)
        else:
            proposal_list = proposals

        # min_global_box_list = self.roi_head.simple_test(min_g_feature, proposal_list,
        #                                                 img_metas, rescale=rescale)
        
        # for idx, res_list in enumerate(min_global_box_list):
        #     #     
        #     resize_bboxes_len5(res_list, scale_min)
        # all_bboxes_lists.append(torch.tensor(min_global_box_list[0][0]))
        # print('all_bboxes_lists[0].shape:',all_bboxes_lists[0].shape)

        # # 2)   
        gaps = [200]
        patch_shape = (1024, 1024)
        p_bs = 1  # patch batchsize
        global_fea_list = []

        for global_shape in gloabl_shape_list:
            scale = img.shape[3]/global_shape[0]
            ratio = global_shape[0]/global_shape_min[0]
            # TODO:est_Concat_Patches_GlobalImg            
            # global_patches_bbox_list, global_full_fea = self.Test_Concat_Patches_GlobalImg(img, ratio, scale,
            #                                                                                min_g_feature,
            #                                                                                patch_shape, gaps, p_bs,
            #                                                                                proposals)
            # all_bboxes_lists.append(global_patches_bbox_list)
        #      
        p_bs=2
        local_bboxes_list = self.Test_Patches_Img(img, patch_shape, gaps, p_bs, proposals, rescale=False)
        # # # ## ��NMS
        all_bboxes_lists.append(local_bboxes_list)
        print(1111111111111111111111111111111111111)
        print(all_bboxes_lists)
        print(all_bboxes_lists.shape)
        bbox_list = merge_results_tensor_hbb(all_bboxes_lists, iou_thr=0.5).cpu()
        print(bbox_list)
        print(bbox_list.shape)
        # bbox_list = merge_results_tensor_hbb(all_bboxes_lists, iou_thr=0.01).cpu()
        final_bbox_list = [bbox_list.numpy()]
        
        # final_bbox_list = [local_bboxes_list.numpy()]
        
        return [final_bbox_list]




    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        #import pdb;pdb.set_trace()
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def onnx_export(self, img, img_metas):

        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        x = self.extract_feat(img)
        proposals = self.rpn_head.onnx_export(x, img_metas)
        if hasattr(self.roi_head, 'onnx_export'):
            return self.roi_head.onnx_export(x, proposals, img_metas)
        else:
            raise NotImplementedError(
                f'{self.__class__.__name__} can not '
                f'be exported to ONNX. Please refer to the '
                f'list of supported models,'
                f'https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html#list-of-supported-models-exportable-to-onnx'  # noqa E501
            )
