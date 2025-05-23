# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrotate.core import rbbox2roi
from ..builder import ROTATED_HEADS
from .rotate_standard_roi_head import RotatedStandardRoIHead
import copy

######
should =   ['ship', 'boat', 'crane', 'goods_yard', 'tank', 'storehouse', 'breakwater', 'dock', 'airplane', 'boarding_bridge', 'runway', 'taxiway', 'terminal', 'apron', 'gas_station', 'truck', 'car', 'truck_parking', 'car_parking', 'bridge', 'cooling_tower', 'chimney', 'vapor', 'smoke', 'genset', 'coal_yard']
ori = ['airplane', 'boat', 'taxiway', 'boarding_bridge', 'tank', 'ship', 'crane', 'car', 'apron', 'dock', 'storehouse', 'goods_yard', 'truck', 'terminal', 'runway', 'breakwater', 'car_parking', 'bridge', 'cooling_tower', 'truck_parking', 'chimney', 'vapor', 'coal_yard', 'genset', 'smoke', 'gas_station']
iidd = [5, 1, 6, 11, 4, 10, 15, 9, 0, 3, 14, 2, 13, 8, 25, 12, 7, 19, 16, 17, 18, 20, 21, 24, 23, 22]
assert [ori[x] for x in iidd] == should
new = [5, 1, 6, 11, 4, 10, 15, 9, 0, 3, 14, 2, 13, 8, 25, 12, 7, 19, 16, 17, 18, 20, 21, 24, 23, 22,26]
######

@ROTATED_HEADS.register_module()
class OrientedStandardRoIHead(RotatedStandardRoIHead):
    """Oriented RCNN roi head including one bbox head."""
    

    def forward_dummy(self, x, proposals):
        """Dummy forward function.

        Args:
            x (list[Tensors]): list of multi-level img features.
            proposals (list[Tensors]): list of region proposals.

        Returns:
            list[Tensors]: list of region of interest.
        """
        outs = ()
        rois = rbbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,flag = False):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task. Always
                set to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if not flag:
            if self.with_bbox:

                num_imgs = len(img_metas)
                if gt_bboxes_ignore is None:
                    gt_bboxes_ignore = [None for _ in range(num_imgs)]
                sampling_results = []
                for i in range(num_imgs):
                    assign_result = self.bbox_assigner.assign(
                        proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                        gt_labels[i])
                    sampling_result = self.bbox_sampler.sample(
                        assign_result,
                        proposal_list[i],
                        gt_bboxes[i],
                        gt_labels[i],
                        feats=[lvl_feat[i][None] for lvl_feat in x])

                    if gt_bboxes[i].numel() == 0:
                        sampling_result.pos_gt_bboxes = gt_bboxes[i].new(
                            (0, gt_bboxes[0].size(-1))).zero_()
                    else:
                        sampling_result.pos_gt_bboxes = \
                            gt_bboxes[i][sampling_result.pos_assigned_gt_inds, :]

                    sampling_results.append(sampling_result)

            losses = dict()
            # bbox head forward and loss
            if self.with_bbox:
                bbox_results = self._bbox_forward_train(x, sampling_results,
                                                        gt_bboxes, gt_labels,
                                                        img_metas)
                losses.update(bbox_results['loss_bbox'])

            return losses
        
        else:

            losses = dict()
            # bbox head forward and loss
            if self.with_bbox:
                bbox_results = self._bbox_forward_train(x, proposal_list,
                                                        gt_bboxes, gt_labels,
                                                        img_metas,flag=flag)
                losses.update(bbox_results['loss_bbox'])

            return bbox_results
        
    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas, flag = False):
        """Run forward function and calculate loss for box head in training.

        Args:
            x (list[Tensor]): list of multi-level img features.
            sampling_results (list[Tensor]): list of sampling results.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        if not flag:
            rois = rbbox2roi([res.bboxes for res in sampling_results])
            bbox_results = self._bbox_forward(x, rois)

            bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                    gt_labels, self.train_cfg)
            loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                            bbox_results['bbox_pred'], rois,
                                            *bbox_targets)

            bbox_results.update(loss_bbox=loss_bbox)
            return bbox_results
        
        else:

            rois = rbbox2roi([res.bbox for res in sampling_results]) ## nums.5的框
            bbox_results = self._bbox_forward(x, rois)
            cls_score_tem = copy.deepcopy(bbox_results['cls_score'])
            
            ####
            cls_score_new = cls_score_tem
            
            bbox_results['cls_score'] = cls_score_new
            #### 

            # bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
            #                                         gt_labels, self.train_cfg)
            loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                            bbox_results['bbox_pred'], rois,
                                            torch.cat(gt_labels,dim = 0).long(),
                                            torch.ones(bbox_results['cls_score'].shape[0]).cuda(),  # tensor([1., 1., 1.,  ..., 1., 1., 1.], device='cuda:0')
                                            None,
                                            None,
                                            flag= flag)

            bbox_results.update(loss_bbox=loss_bbox)
            return bbox_results



    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False,
                           large = False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains \
                the boxes of the corresponding image in a batch, each \
                tensor has the shape (num_boxes, 5) and last dimension \
                5 represent (cx, cy, w, h, a, score). Each Tensor \
                in the second list is the labels with shape (num_boxes, ). \
                The length of both lists should be equal to batch_size.
        """

        rois = rbbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        selcts = []
        select_cls_score = []
        all_box_cls = []
        if large:
            for i in range(len(proposals)):
                det_bbox, det_label,selct,box_cls = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg,large = large)
                det_bboxes.append(det_bbox)
                det_labels.append(det_label)
                selcts.append(selct)
                all_box_cls.append(box_cls)
            
            #### 获取分类分数
            for kk in range(len(selcts)):
                select_cls_score.append(cls_score[kk][selcts[kk],:])
            ####
            return det_bboxes, det_labels, select_cls_score,all_box_cls  ## check is right
            ### check
                    #### check det_labels, selec_cls_scores
            # for y1,y2 in zip(det_labels, select_cls_score):
            #     for yy1,yy2 in zip(y1,y2):
            #         if not int(np.argmax(np.array(yy2.cpu())[:-1])) == int(yy1):
            #             print(0)
            #             0
            #             0

        
        else:
            for i in range(len(proposals)):
                det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg,large = large)
                det_bboxes.append(det_bbox)
                det_labels.append(det_label)
            return det_bboxes, det_labels
