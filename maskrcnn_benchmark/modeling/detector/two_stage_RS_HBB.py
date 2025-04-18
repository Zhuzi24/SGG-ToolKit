# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import torch
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
import random
from mmdet.utils import get_device
from .img_split_bridge_tools_hbb import *

import numpy as np
import torch.nn.functional as F
import copy
import torch.nn as nn
from mmdet.core.bbox.iou_calculators import bbox_overlaps


pos_HBB =  [48, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]

def hs(pathches_cls_scores, p_keeps):
    all_cls = [item for sublist in pathches_cls_scores for item in sublist]
    
    mer_cls_scores = []
    for cls_id in np.arange(48):
        cls_part = [all_cls[pp][cls_id].reshape((1, 49)) if all_cls[pp][cls_id].ndim == 1 
                    else all_cls[pp][cls_id].reshape((1,49,4)) if all_cls[pp][cls_id].shape == (49, 4) 
                    else all_cls[pp][cls_id] 
                    for pp in np.arange(len(all_cls)) if len(all_cls[pp][cls_id]) != 0]
        mer_cls_scores.append(np.concatenate(cls_part, axis=0) if cls_part else [])
        
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
                    else all_scores[pp][cls_id].reshape((1,49,4)) if all_scores[pp][cls_id].shape == (49, 4) 
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


def resize_bboxes_len5(bboxes_out,each_class,scale):
    """Resize bounding boxes with scales."""

    for i in range(len(bboxes_out)):
        box_out=bboxes_out[i]
        each = each_class[i]
        w_scale = scale
        h_scale = scale
        box_out[:, 0] *= w_scale
        box_out[:, 1] *= h_scale
        box_out[:, 2] *= w_scale
        box_out[:, 3] *= h_scale

        if len(each) != 0:
            each[:,:, 0] *= w_scale
            each[:,:, 1] *= h_scale
            each[:,:, 2] *= w_scale
            each[:,:, 3] *= h_scale


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
            info['width'] = img.shape[2]
            info['height'] = img.shape[1]

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
        info['width'] = img.shape[2]
        info['height'] = img.shape[1]

        sizes = [patch_shape[0]]
        # gaps=[0]
        windows = get_sliding_window(info, sizes, gaps, img_rate_thr)
        patchs, patch_infos = crop_img_withoutann(info, windows, img,
                                                  no_padding=False,
                                                  padding_value=[0.0081917211329, -0.004901960784, 0.0055655449953] )

       
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

def relocate(idx, local_bboxes,local_each, patch_meta,flag_re = False):
    
    # local_boxes:(n,5):(x0,y0,x1,y1,score)
    # put patches' local bboxes to full img via patch_meta
    meta = patch_meta[idx]
    top = meta['y_start']
    left = meta['x_start']

    # local_bboxes_tmp = local_bboxes[0]
    # local_bboxes_tmp[:, 0] += left
    # local_bboxes_tmp[:, 1] += top
    # local_bboxes_tmp[:, 2] += left
    # local_bboxes_tmp[:, 3] += top

    for in_class in range(len(local_bboxes)):
        local_bboxes_tmp = local_bboxes[in_class]
        each_class = local_each[in_class]

        if len(local_bboxes_tmp) != 0:
            local_bboxes_tmp[:, 0] += float(left)
            local_bboxes_tmp[:, 1] += float(top)
            local_bboxes_tmp[:, 2] += float(left)
            local_bboxes_tmp[:, 3] += float(top)


        if len(each_class) != 0: 
          each_class[:, :, 0] += float(left)
          each_class[:, :, 1] += float(top)           
          each_class[:, :, 2] += float(left)
          each_class[:, :, 3] += float(top)     

        # for i in range(len(local_bboxes_tmp)):
        #     bbox = local_bboxes_tmp[i]
        #     # print('local_bboxes[i]:',bbox)
        #     bbox[0] += left
        #     bbox[1] += top
        #     bbox[2] += left
        #     bbox[3] += top

        #     each_class_bbox =  each_class[i]
        #     each_class_bbox[:,0] += left
        #     each_class_bbox[:,1] += top
        #     each_class_bbox[:,2] += left
        #     each_class_bbox[:,3] += top
       
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
class TwoStageDetector_RS_HBB(nn.Module):
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
                 init_cfg=None,
                 ori_cfg= None):
        super(TwoStageDetector_RS_HBB, self).__init__()
        self.version = 'le90'
 
        ori_cfg  =  ori_cfg if  ori_cfg else init_cfg
        self.ori_cfg = ori_cfg if  ori_cfg else init_cfg

        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        self.backbone_d2 = build_backbone(backbone)
    

        if neck is not None:
            self.neck = build_neck(neck)
            self.neck_d2 = build_neck(neck)
          


        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)
            self.rpn_head_d2 = build_head(rpn_head_)
    

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)
            self.roi_head_d2 = build_head(roi_head)



        from maskrcnn_benchmark.modeling.roi_heads.roi_heads import build_roi_heads
        self.roi_heads = build_roi_heads(self.ori_cfg, self.ori_cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS)

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
        x = self.backbone(img.cuda())
        x = self.neck(x)
        return x

    def extract_feat_d2(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone_d2(img.cuda()) ### 1220
    
        x = self.neck_d2(x)
        return x
    

    def Test_Patches_Img(self,img,patch_shape,gaps, p_bs, proposals, rescale=False):
        # Crop full img into patches
        gt_bboxes=[]
        gt_labels=[]
        local_bboxes_lists=[]

        pathches_cls_scores = []
        l_all_box_cls = []

        p_imgs, p_metas = FullImageCrop(self, img, gt_bboxes, gt_labels,
                                        patch_shape=patch_shape,
                                        gaps=gaps, mode='test')

        # p_bs = 1
        temp = img.shape[0]
        H =  img.shape[2]
        W = img.shape[3]

        img = img.cpu()
        del img


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
                    patch=patch.cuda()
                    x = self.extract_feat(patch)
                    if proposals is None:
                        proposal_list = self.rpn_head.simple_test_rpn(x, patch_meta)
                    else:
                        proposal_list = proposals
                                      
                    
                                     
                    local_bbox_list, l_selec_cls_scores, new_all_box_cls = self.roi_head.simple_test(
                        x, proposal_list, patch_meta, rescale=rescale, large = True)


                    for idx, (res_list,each_box) in enumerate(zip(local_bbox_list,new_all_box_cls)):
                        det_bboxes = res_list
                        relocate(idx, det_bboxes,each_box, patch_meta)


                    local_bboxes_lists.append(local_bbox_list)
                    pathches_cls_scores.append(l_selec_cls_scores)
                    l_all_box_cls.append(new_all_box_cls)

                   

                j = j+p_bs


        torch.cuda.empty_cache()
        x = tuple(tensor.cpu() for tensor in x)
        del x

       
        bbox_list, p_keeps  = merge_results_two_stage_hbb(local_bboxes_lists,iou_thr=0.4,flag = 2)
        
        new_mer_cls_scores = hs(pathches_cls_scores,p_keeps)
        new_l_all_box_cls = hs(l_all_box_cls,p_keeps)   


        out_list = [tt if tt.shape[-1] == 5 else np.zeros((0, 5)) for tt in bbox_list]

        return out_list, new_mer_cls_scores, new_l_all_box_cls


    def Test_Concat_Patches_GlobalImg(self, ori_img, ratio, scale, g_fea, patch_shape, gaps, p_bs, proposals, rescale=False,id = None):
        
        device=get_device()

        if (ori_img.shape[2] > 10000 or ori_img.shape[3] > 10000):
            ori_img_cpu = ori_img.cpu()
          
            img = F.interpolate(ori_img_cpu, scale_factor=1 / scale, mode='bilinear')
            img = img.to(device) 
           
        else:
            img = F.interpolate(ori_img, scale_factor=1 / scale, mode='bilinear')


            
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
       
            while j < len(p_imgs[i]):
                if (j + p_bs) >= len(p_imgs[i]):
                    patch = patches[j:]
                    patch_meta = patches_meta[j:]
                else:
                    patch = patches[j:j + p_bs]
                    patch_meta = patches_meta[j:j + p_bs]  # x_start and y_start
              
                with torch.no_grad():
                  
                    patch=patch.to(device)
                    patch_fea = self.all_extract_feat[id](patch)

                    if proposals is None:
                        proposal_list = self.all_RPN[id].simple_test_rpn(patch_fea, patch_meta)
                    else:
                        proposal_list = proposals
                    
                    global_bbox_list,g_selec_cls_scores, new_all_box_cls  = self.all_ROI[id].simple_test(
                        patch_fea, proposal_list, patch_meta, rescale=rescale,large = True)
                    


                    for idx, (res_list,res_list_each_class) in enumerate(zip(global_bbox_list,new_all_box_cls)):
                                           
                        relocate(idx, res_list,res_list_each_class, patch_meta)    
                        resize_bboxes_len5(res_list,res_list_each_class, scale)


                    patches_bboxes_lists.append(global_bbox_list)
                    pathches_cls_scores.append(g_selec_cls_scores)
                    g_all_box_cls.append(new_all_box_cls)

                    torch.cuda.empty_cache()
                    patch = patch.cpu()
                    patch_fea = tuple(tensor.cpu() for tensor in patch_fea)
                    del patch_fea, patch


                    conf_thr=0.4
                    conf_thr=0
                    num_thr=0  # 数量阈值,global下检测到多于num_thr的则精细检测 
                    box_count=np.sum(res_list[0][:,-1]>conf_thr)
                    
                    if box_count>num_thr and scale==2: #只在down2阶段进行加速,更高层金字塔可能会不准确

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
                             
                                padd=torch.tensor(padd.transpose((2,1,0)),
                                                        device=sub_img.device)
                                
                                padd[ ...,:sub_img.shape[1], :sub_img.shape[2]] = sub_img
                                sub_img = padd
            
                            
                            
                       
                            sub_img_list.append(sub_img)
                        
              
                            
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

                        
                        for idx, (res_list,sub_class) in enumerate(zip(sub_bbox_list,sub_all_box_cls)):
                                                        
                            relocate(idx, res_list,sub_class, sub_meta_list)
                        
                   
                        patches_bboxes_lists.append(sub_bbox_list)
                        pathches_cls_scores.append(sub_selec_cls_scores)
                        g_all_box_cls.append(sub_all_box_cls)

                j = j + p_bs

       
        patches_bboxes_list,p_keeps = merge_results_two_stage_hbb(patches_bboxes_lists, iou_thr=0.4,flag=1)

        new_mer_cls_scores = hs(pathches_cls_scores,p_keeps)
        new_g_all_box_cls = hs(g_all_box_cls,p_keeps) 
        

        out_list = [tt if tt.shape[-1] == 5 else np.zeros((0, 5)) for tt in patches_bboxes_list]

        full_patches_out =[]
        return out_list, full_patches_out, new_mer_cls_scores,new_g_all_box_cls


    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""

        all_bboxes_lists=[]
        global_shape_h = img.shape[2]
        global_shape_w = img.shape[3]

        if global_shape_h > 10000 or global_shape_w > 10000:
            p_bs  = 1
            p_bs_2 = 1
        else:
            p_bs = 4
            p_bs_2 = 2


        global_shape_max=max(global_shape_h,global_shape_w)


        gaps = [200]
        patch_shape = (1024, 1024)
        
        if global_shape_max <= 1024:  # all

            local_bboxes_list,local_each_cls_scores,l_box_en= self.Test_Patches_Img(img, patch_shape, gaps, p_bs_2, proposals, rescale=False)

            local_bboxes = [local_bboxes_list] 

            all =  local_bboxes
            #### 合并
            all_scores =   [local_each_cls_scores]
            all_en =  [l_box_en]

            all_nms, all_keeps = merge_results_two_stage_hbb(all, iou_thr=0.4,flag=3)
            new_mer_cls_scores = hs_all(all_scores,all_keeps)
            new_en = hs_all(all_en,all_keeps)   

        else:

            # (2) 按比例进行大图推理 
            gloabl_shape_list=[]
            while global_shape_max > 1024:
                global_shape_h =  global_shape_h/2
                global_shape_w =  global_shape_w/2
                global_shape_max = global_shape_max/2

                gloabl_shape_list.append((global_shape_h, global_shape_w))
                
            global_shape_min = (global_shape_h,global_shape_w)
          
            global_fea_list = []
            global_each_cls_scores = []
            g_box_en = []

            level= 0
            for global_shape in gloabl_shape_list:
                scale = img.shape[3]/global_shape[1]
                ratio = global_shape[0]/global_shape_min[0]
                # TODO:est_Concat_Patches_GlobalImg            
                global_patches_bbox_list, global_full_fea, each_cls_scores,each_box_en  = self.Test_Concat_Patches_GlobalImg(img, ratio, scale,
                                                                                            None,
                                                                                            patch_shape, gaps, p_bs,
                                                                                            proposals,id = level)
                all_bboxes_lists.append(global_patches_bbox_list)
                global_each_cls_scores.append(each_cls_scores)
                g_box_en.append(each_box_en)
                level = 1 


            all = all_bboxes_lists

            #### 合并
            all_scores =  global_each_cls_scores 
            all_en = g_box_en

            all_nms, all_keeps = merge_results_two_stage_hbb(all, iou_thr=0.4,flag=3)
            new_mer_cls_scores = hs_all(all_scores,all_keeps)
            new_en = hs_all(all_en,all_keeps)               
            
            

        
        
        all_nms_list = [tt if tt.shape[-1] == 5 else np.zeros((0, 5)) for tt in all_nms]
        
        return [all_nms_list] ,new_mer_cls_scores,new_en
    
    def find_zero_area_boxes_optimized(self,bbox):
        areas = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
        zero_area_boxes_indices = torch.nonzero(areas == 0, as_tuple=True)[0].tolist()
        return zero_area_boxes_indices

    def batch(self,img,targets,ite=None):
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
        
            for img, img_meta in zip(imgs, img_metas):
                batch_size = len(img_meta)
                for img_id in range(batch_size):
                    
                    img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])
  
            if num_augs == 1:
                results, cls_scores,new_en = self.simple_test(imgs[0], img_metas[0])


            ### 
            sclec_id = []
            f_results = []
            f_cls_scores = []
            f_en = []
            for k1 in range(len(results[0])):
                sclec_id.append([])
                get_data = results[0][k1]
                cls = cls_scores[k1]
                conf = get_data[:,4]
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
                    f_cls_scores.append(cls[pos])
                    sclec_id[k1].append(pos)
            
            ##  合并所有bbox
            no_f_results = [f for f in f_results if len(f) != 0]
           
            # check empty bbox
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
                    conf = get_data[:,4]
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
                        get_data = results[0][k1]
                        cls1 = cls_scores[k1]
                        conf = get_data[:,4]
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
                            get_data = results[0][k1]
                            cls1 = cls_scores[k1]
                            conf = get_data[:,4]
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
                                get_data = results[0][k1]
                                cls1 = cls_scores[k1]
                                conf = get_data[:,4]
                                en = new_en[k1]
                                pos =  np.where(conf >= 0.000001)[0]
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
            all_score = all_score[:,pos_HBB]

            proposals = copy.deepcopy(targets[0])



            zero_list = self.find_zero_area_boxes_optimized(all_box[:,:4])
            if len(zero_list) != 0:
                print("exist zero area boxes")
                N = list(range(len(all_box)))
                N_s = [x for x in N if x not in zero_list]  
                all_box = all_box[N_s]
                all_score = all_score[N_s]
                all_en = all_en[N_s]



            ####
            ### 异常值检测
            proposals.bbox = all_box[:,:4]



            proposals.extra_fields["predict_logits"] = all_score
            proposals.extra_fields["boxes_per_cls"] = all_en
            assert len(all_en) == len(all_score)
            del proposals.extra_fields["labels"]
            
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



            iou = torch.tensor(bbox_overlaps(  # 533,1003
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
            

            b1 = targets[0].bbox

        
           

            b2 = targets[0].extra_fields["target1"].bbox

            
            if 0 in b2[:,0] or 0 in b1[:,0]:
                 x_nonnan_indices = torch.nonzero(~torch.isnan(b2[:,0] / b1[:,0]) & (b2[:,0] / b1[:,0] != 0), as_tuple=False)
                 x_first_nonnan_index = x_nonnan_indices[0, 0].item() if x_nonnan_indices.numel() > 0 else None

                  
                 w_f = float((b2[:,0] / b1[:,0])[x_first_nonnan_index])

            else:
                    w_f = float((b2[:,0] / b1[:,0])[0])


            if 0 in b2[:,1] or 0 in b1[:,1]:
                 
                 y_nonnan_indices = torch.nonzero(~torch.isnan(b2[:,1] / b1[:,1]) & (b2[:,1] / b1[:,1] != 0), as_tuple=False)
                 y_first_nonnan_index = y_nonnan_indices[0, 0].item() if y_nonnan_indices.numel() > 0 else None
                 h_f = float((b2[:,1] / b1[:,1])[y_first_nonnan_index])
            
            else:
                  
                 h_f = float((b2[:,1] / b1[:,1])[0])


      
            if self.training:
                if isinstance(targets[0].extra_fields["data1"]["img_metas"],list):
                    s_size = targets[0].extra_fields["data1"]["img_metas"][0].data["pad_shape"]
                else:
                    s_size = targets[0].extra_fields["data1"]["img_metas"].data["pad_shape"]
            else:
                if isinstance(targets[0].extra_fields["data1"]["img_metas"],list):
                     s_size = targets[0].extra_fields["data1"]["img_metas"][0].data["pad_shape"]
                else:
                     s_size = targets[0].extra_fields["data1"]["img_metas"].data["pad_shape"]
            sh,sw =  s_size[0],s_size[1]
            
            proposals.bbox[:,0] *= w_f
            proposals.bbox[:,1] *= h_f
            proposals.bbox[:,2] *= w_f
            proposals.bbox[:,3] *= h_f

            proposals.size = (sw,sh)

            iou2 = torch.tensor(bbox_overlaps(  # 533,1003
                b2.float(),
                proposals.bbox.float()).cpu().numpy()).cuda()
            
            if not self.training:
                return proposals,[w_f,h_f]
            return proposals,None
    


    def forward(self, img, targets=None, logger=None, ite=None,  gt_bboxes_ignore=None, gt_masks=None, proposals=None, sgd_data = None, m = None, val = None,
                vae = None,  **kwargs): 
        

        imgs = img.tensors
        if self.tasks == "Predcls":  ###  given Object Detection
            losses = dict()
            x = self.extract_feat(imgs) # feature
            proposals = targets
            if self.roi_heads:  ### relation
                 if self.ori_cfg.CFA_pre == 'extract_aug':
                    tail_dict = self.roi_heads(x, proposals, targets, logger,ite=ite,OBj = self.roi_head)
                    return tail_dict
                 x, result, detector_losses = self.roi_heads(x, proposals, targets, logger,
                                                        ite=ite,OBj = self.roi_head, m=m, val = val,vae = vae)
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
                     gt_bboxes.append(tar.extra_fields["data"]["gt_bboxes"].data.float().cuda() if self.training else tar.extra_fields["data"]["gt_bboxes"][0].data.float().cuda())
                     gt_labels.append(tar.extra_fields["data"]["gt_labels"].data.long().cuda() if self.training else tar.extra_fields["data"]["gt_labels"][0].data.long().cuda())  
            else:
                img_metas =  [ targets[0].extra_fields["data"]["img_metas"].data] if self.training else [ targets[0].extra_fields["data"]["img_metas"][0].data]

                if self.ori_cfg.CFA_pre == "extract_aug":
                    gt_bboxes = [ targets[0].extra_fields["data"]["gt_bboxes"].data.float().cuda()] 
                    gt_labels = [ targets[0].extra_fields["data"]["gt_labels"][0].data.long().cuda()]
                else:
                    gt_bboxes = [ targets[0].extra_fields["data"]["gt_bboxes"].data.float().cuda()] if self.training else [ torch.tensor(targets[0].extra_fields["data"]["gt_bboxes"][0].data).float().cuda() ]
                    gt_labels = [ targets[0].extra_fields["data"]["gt_labels"].data.long().cuda()] if self.training else [ torch.tensor(targets[0].extra_fields["data"]["gt_labels"][0].data).long().cuda() ]
            losses = dict()
            x = self.extract_feat(imgs) # feature
            proposals = targets     
            ## no position change  bbox_results['cls_score'] =  cls_score_tem 
            bbox_results = self.roi_head.forward_train(x, img_metas, proposals,
                                                    gt_bboxes, gt_labels,
                                                    gt_bboxes_ignore, gt_masks, flag = True,
                                                         **kwargs)
            ## cls_score
            cls_score = bbox_results["cls_score"][:,pos_HBB]
            start = 0    
            for pro in proposals:
                lens = len(pro)
                pro.extra_fields["predict_logits"] = cls_score[start : start+lens,:]
                start = lens


            if self.ori_cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "GCN_RELATION" or "HetSGG_Predictor":
    
                for pro in proposals: 
                  
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
                 if self.ori_cfg.CFA_pre == 'extract_aug':
                    tail_dict = self.roi_heads(x, proposals, targets, logger,ite=ite,OBj = self.roi_head)
                    return tail_dict
                 
                 x, result, detector_losses = self.roi_heads(x, proposals, targets, logger,
                                                        ite=ite,OBj = self.roi_head, m=m,val = val,vae = vae)

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

            if self.roi_heads:  ### relation
                 x, result, detector_losses = self.roi_heads(x, p, sgd_data[1], logger,
                                                        ite=ite,OBj = self.roi_head,s_f = s_f) 

            if self.training:     
                losses.update(detector_losses)
                return losses 
            else:
                return result