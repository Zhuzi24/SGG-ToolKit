# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
import numpy as np 
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_union, boxlist_intersection
from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
from .sampling import make_roi_box_samp_processor
import copy
import torch.nn.functional as F
def add_predict_logits(proposals, class_logits):
    slice_idxs = [0]
    for i in range(len(proposals)):
        slice_idxs.append(len(proposals[i])+slice_idxs[-1])
        proposals[i].add_field("predict_logits", class_logits[slice_idxs[i]:slice_idxs[i+1]])
    return proposals

class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, half_out=self.cfg.MODEL.ATTRIBUTE_ON)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)
        self.samp_processor = make_roi_box_samp_processor(cfg)
        self.celoss = nn.CrossEntropyLoss()
    
    def sel_prop(self,Prop,id):
        
            assert len(Prop) == len(id)
            
         
            
            del_id = []
            re_id = []
            # else:
            for j in range(len(Prop)):
                del_id.append([])
            # for p1,id1 in zip(Prop,id):
                box1 = Prop[j].bbox.tolist()
                new_box = [box1[x] for x in id[j]]
                
                #####再次筛选
                for k in range(len(new_box)):
                    if new_box[k].count(0) >2:
                        del_id[j].append(k) 
                    elif new_box[k].count(0)==2:
                        if not (new_box[0]==0 and new_box[1]==0):
                              del_id[j].append(k) 
                    else: 
                        if len(self.find_equal_values_and_positions(new_box[k])) != 0 :
                            del_id[j].append(k) 
                #####
                new_box1 =  [new_box[n] for n in range(len(new_box)) if n not in del_id[j]] #[new_box[n] if n not in del_id[j] else "" for n in range(len(new_box))]
                assert len(new_box) == len(new_box1) + len(del_id[j])
                Prop[j].bbox = torch.Tensor(new_box1).cuda()
    
                my_list =  id[j]

                indices_to_remove = del_id[j]
                new_list = [value for index, value in enumerate(my_list) if index not in indices_to_remove]
                re_id.append(new_list)

            
            return Prop,re_id
        
    def calculate_iou(self,box1, box2):
            # 计算两个框的交集部分的坐标
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])

            # 计算交集的面积
            intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

            # 计算两个框各自的面积
            area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
            area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

            # 计算并集的面积
            union_area = area_box1 + area_box2 - intersection_area

            # 计算交并比
            iou = intersection_area / union_area

            return iou
        
    def find_equal_values_and_positions(self,lst):
        positions = {}
        
        for i, val in enumerate(lst):
            if val in positions:
                positions[val].append(i)
            else:
                positions[val] = [i]

        # 返回具有相等值的位置
        return {key: value for key, value in positions.items() if len(value) > 1}


    def find_nearest_boxes(self,boxes,ite = None,logger = None):
        nearest_boxes = []
        
        for i, box1 in enumerate(boxes):
            

            
            iou_candidates = []

            # 查找与该框IOU>0的框
            for j, box2 in enumerate(boxes):
                if i != j:
                    iou = self.calculate_iou(box1, box2)
                    if iou > 0:
                        iou_candidates.append((j, iou))

            # 如果满足IOU的框>=2个，返回IOU最大的两个框的位置id
            if len(iou_candidates) >= 2:
                iou_candidates.sort(key=lambda x: x[1], reverse=True)
                nearest_boxes.append((i, iou_candidates[0][0], iou_candidates[1][0]))
                
            elif len(iou_candidates) == 1:
                # 如果IOU的框只有1个，记录该框的位置id，继续寻找另一个框
                # first_candidate = iou_candidates[0]
                # nearest_boxes.append((i, first_candidate[0]))
                
                iou_candidates = []
                # 寻找另一个框的位置id
                for scale1 in [1, 2, 5, 10, 12, 15, 20,50,100,500]:
                    scaled_box = [
                        box1[0] - (box1[2] - box1[0]) * (scale1 - 1) / 2,
                        box1[1] - (box1[3] - box1[1]) * (scale1 - 1) / 2,
                        box1[2] + (box1[2] - box1[0]) * (scale1 - 1) / 2,
                        box1[3] + (box1[3] - box1[1]) * (scale1 - 1) / 2
                    ]

                   
                    for j, box2 in enumerate(boxes):
                        if i != j and self.calculate_iou(scaled_box, box2) > 0:
                            if j not in iou_candidates:
                                iou_candidates.append(j)

                    if len(iou_candidates) >= 2:
                        nearest_boxes.append((i, iou_candidates[0], iou_candidates[1]))
                        break
                    
                
                # for j, box2 in enumerate(boxes):
                #     if i != j and self.calculate_iou(box1, box2) > 0:
                #         nearest_boxes[-1] = (i, first_candidate[0], j)
                #         break
            else:
                # 如果满足IOU的框为0个，将该框逐渐放大方法，直到出现有与其iou》0的前2个框，记录id，返回
                iou_candidates = []
                for scale in [2, 5, 10, 12, 15, 20,50,100,500]:
                    scaled_box = [
                        box1[0] - (box1[2] - box1[0]) * (scale - 1) / 2,
                        box1[1] - (box1[3] - box1[1]) * (scale - 1) / 2,
                        box1[2] + (box1[2] - box1[0]) * (scale - 1) / 2,
                        box1[3] + (box1[3] - box1[1]) * (scale - 1) / 2
                    ]

                    
                    for j, box2 in enumerate(boxes):
                        if i != j and self.calculate_iou(scaled_box, box2) > 0:
                             if j not in iou_candidates:
                                     iou_candidates.append(j)

                    if len(iou_candidates) >= 2:
                        nearest_boxes.append((i, iou_candidates[0], iou_candidates[1]))
                        break
        # if len(nearest_boxes) != len(boxes):
            #t = 1
            if len(nearest_boxes) != i+1:
                if i == 0:
                    nearest_boxes.append((0, 1, 2))
                    logger.info("all !!!")
                    logger.info(box1)
                else:
                
                  nearest_boxes.append(nearest_boxes[-1])
        
        return nearest_boxes

    def sel_uni_id(self,fliter_box_prop,ite = None,logger = None):
        uni_id = []
        for f1 in fliter_box_prop:
            # if len(f1)==174  and ite == 2:
            #     t=1
            lis = self.find_nearest_boxes(f1.bbox.tolist(),ite = ite,logger = logger)
            assert len(lis) == len(f1)
            uni_id.append(lis)
        
        
        return uni_id
        
        
        
        

    def forward(self, features, proposals, targets=None,m=None,GLO_f = None,iteration = None,val = None,logger = None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        ###################################################################
        # box head specifically for relation prediction model
        ###################################################################
        if self.cfg.MODEL.RELATION_ON:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
                # use ground truth box as proposals
                proposals = [target.copy_with_fields(["labels", "attributes"]) for target in targets]
                x = self.feature_extractor(features, proposals)
                if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                    # mode==predcls
                    # return gt proposals and no loss even during training
                    return x, proposals, {}
                else:
                    # mode==sgcls
                    # add field:class_logits into gt proposals, note field:labels is still gt
                    class_logits, _ ,cls_new = self.predictor(x)
                    proposals = add_predict_logits(proposals, class_logits)
                    # proposals = add_predict_logits(proposals, class_logits,cls_new)
                    return x, proposals, {} 
            else:
                # mode==sgdet
                if self.training or not self.cfg.TEST.CUSTUM_EVAL:
                    proposals = self.samp_processor.assign_label_to_proposals(proposals, targets)
                x = self.feature_extractor(features, proposals)
                class_logits, box_regression = self.predictor(x)
                proposals = add_predict_logits(proposals, class_logits)
                # post process:
                # filter proposals using nms, keep original bbox, add a field 'boxes_per_cls' of size (#nms, #cls, 4)
                x, result,_ = self.post_processor((x, class_logits, box_regression), proposals, relation_mode=True)
                # note x is not matched with processed_proposals, so sharing x is not permitted
                return x, result, {}

        #####################################################################
        # Original box head (relation_on = False)
        #####################################################################
        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
               # copy_pro = copy.deepcopy(proposals)
                
                proposals = self.samp_processor.subsample(proposals, targets)
                copy_pro = copy.deepcopy(proposals)
                
                ####
               # after_copy_pro = self.samp_processor.assign_label_to_proposals(copy_pro, targets)
               
        copy_pro = copy.deepcopy(proposals)
        #############
        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x,_= self.feature_extractor(features, proposals)
       
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)
        
        
        ########### 11 24 
        
        #union_vis_features = self.feature_extractor.pooler(x, union_proposals)

        # val  =2 
        if m is not None:
            if val is None:
                
                _, _, save_id = self.post_processor((x, class_logits, box_regression), proposals)
                delcpoy_id = [list(set(id1.tolist())) for id1 in save_id]
                delcpoy_id.sort()
                fliter_box_prop,new_id = self.sel_prop(proposals,delcpoy_id)
                delcpoy_id = new_id
                #####  重新计算
                x,_= self.feature_extractor(features, fliter_box_prop)
                class_logits, box_regression = self.predictor(x)
                
                labels = []
                for xx in range(len(copy_pro)):
                    labels.append(copy_pro[xx].extra_fields["labels"])
                after_labels = []
                for id_tem,ll in zip(delcpoy_id,labels):
                    after_labels.append(ll[id_tem])
                after_labels = torch.cat(after_labels,dim=0)
                ######
                
                union_id = self.sel_uni_id(fliter_box_prop,ite = iteration,logger = logger)
                union_proposals = []
                for proposal, rel_pair_idx in zip(fliter_box_prop, union_id):
                    rel_pair_idx = torch.Tensor(rel_pair_idx).long().cuda()
                    head_proposal = proposal[rel_pair_idx[:, 0]]
                    mid_proposal = proposal[rel_pair_idx[:, 1]]
                    tail_proposal = proposal[rel_pair_idx[:, 2]]
                    t = 1
                    union_proposal = boxlist_union(head_proposal,tail_proposal,flag=2, boxlist_mid=mid_proposal)
                    union_proposals.append(union_proposal)
           
                _,uni_fea= self.feature_extractor(features, union_proposals)
        
                rel_tem = copy.deepcopy(class_logits)
                loss_mse, gen = m.Bias(G = uni_fea, inputdata=rel_tem, iteration=iteration, ff=after_labels, val=val, logger=logger)
                return x, proposals, dict(loss_classifier=1,loss_mse = 1)
                # class_logits = class_logits + gen
        
        
        
                ####################
                # _, _, save_id = self.post_processor((x, class_logits, box_regression), proposals)
                # #### 117 修改
                # delcpoy_id = [list(set(id1.tolist())) for id1 in save_id]
                # class_logits_split = torch.chunk(class_logits, 16, dim=0)
                # # assert sum(class_logits_split[1][0]) == sum(class_logits[256])
                # box_regression_split = torch.chunk(box_regression, 16, dim=0)
                # # assert sum(box_regression_split[1][0]) == sum(box_regression[256])
                # tem_split = torch.chunk(tem, 16, dim=0)
                # # assert sum(sum(sum(tem_split[1][0]))) == sum(sum(sum(tem[256])))

                # labels = []
                # for xx in range(len(proposals)):
                #     labels.append(proposals[xx].extra_fields["labels"])

                # after_class_logits_split = []
                # after_box_regression_split = []
                # # after_tem_split = []
                # after_labels = []
                # for id_tem, cla_tem, box_tem, ttt, ll in zip(delcpoy_id, class_logits_split, box_regression_split,
                #                                              tem_split, labels):
                #     after_class_logits_split.append(cla_tem[id_tem, :])
                #     after_box_regression_split.append(box_tem[id_tem, :])
                #     after_tem_split.append(ttt[id_tem, :, :, :])
                #     after_labels.append(ll[id_tem])

                # class_logits, box_regression, tem, labels = torch.cat(after_class_logits_split,dim = 0 ), torch.cat(
                #     after_box_regression_split,dim = 0), \
                #     torch.cat(after_tem_split,dim = 0), torch.cat(after_labels,dim = 0)

                # ####
                # rel_tem = copy.deepcopy(class_logits.detach())
                # #####
                # num_rels = []
                # for n_tem in  delcpoy_id :
                #     num_rels.append(len(n_tem))

                # S = torch.empty((sum(num_rels), GLO_f.shape[1], GLO_f.shape[2], GLO_f.shape[3])).cuda()
                # start_idx = 0
                # for i, count in enumerate(num_rels):
                #     end_idx = start_idx + count
                #     S[start_idx:end_idx, :, :, :] = GLO_f[i, :, :, :].unsqueeze(0).repeat(count, 1, 1, 1)
                #     start_idx = end_idx

                # # assert sum(sum(sum(S[0])) == sum(sum(S[num_rels[0] - 1])))



                # # label_tem = copy.deepcopy(torch.cat(targets, -1).detach())
                # # gen = m.Bias(S, uni_tem, val=val, logger=logger)
                # loss_mse, gen = m.Bias(G = S,L = tem, inputdata=rel_tem, iteration=iteration, ff=labels, val=val, logger=logger)
                # class_logits = class_logits + gen

                # loss_classifier = self.celoss(class_logits,labels.long())
                # #loss_classifier, loss_box_reg = self.loss_evaluator([class_logits], [box_regression], proposals)

                # return x, proposals, dict(loss_classifier=loss_classifier,loss_mse =loss_mse)
                #################

            else:
                ################ 1124
                # rel_tem = copy.deepcopy(class_logits.detach())
                # if len(rel_tem ) == 2000:
                #     num_rels = [2000]
                #     S = torch.empty(2000, GLO_f.shape[1], GLO_f.shape[2], GLO_f.shape[3]).cuda()
                #     S[:,:,:,:] = GLO_f[0]
                #     gen = m.Bias(G=S, L=tem, inputdata=rel_tem, ff=None, val=val,
                #                  logger=logger)
                #     class_logits = class_logits + gen

                # else:


                #     num_rels = [2000] * 2
                #    # for n_tem in save_id:
                #    #     num_rels.append(len(n_tem))

                #     S = torch.empty((sum(num_rels), GLO_f.shape[1], GLO_f.shape[2], GLO_f.shape[3])).cuda()
                #     start_idx = 0
                #     for i, count in enumerate(num_rels):
                #        end_idx = start_idx + count
                #        S[start_idx:end_idx, :, :, :] = GLO_f[i, :, :, :].unsqueeze(0).repeat(count, 1, 1, 1)
                #        start_idx = end_idx

                #     gen = m.Bias(G=S, L=tem, inputdata=rel_tem, ff=None, val=val,
                #                           logger=logger)
                #     class_logits = class_logits + gen
                ####################### 11 24 

                #####################
                # num_rels = []
                # for n_tem in save_id:
                #     num_rels.append(len(n_tem))
                #
                # S = torch.empty((sum(num_rels), GLO_f.shape[1], GLO_f.shape[2], GLO_f.shape[3])).cuda()
                # start_idx = 0
                # for i, count in enumerate(num_rels):
                #     end_idx = start_idx + count
                #     S[start_idx:end_idx, :, :, :] = GLO_f[i, :, :, :].unsqueeze(0).repeat(count, 1, 1, 1)
                #     start_idx = end_idx

                # assert sum(sum(sum(S[0])) == sum(sum(S[num_rels[0] - 1])))


                # class_logits = class_logits + gen
                # x_bf = copy.deepcopy(x)
                # class_logits_bf = copy.deepcopy(class_logits)
                # box_regression_bf = copy.deepcopy(box_regression)
                # proposals_bf = copy.deepcopy(proposals)
                #
                # _, _,save_id = self.post_processor((x_bf, class_logits_bf, box_regression_bf), proposals_bf)
                # delcpoy_id = [list(set(id1.tolist())) for id1 in save_id]
                #
                # ##########
                # if len(delcpoy_id) == 1:
                #     after_class_logits_split = class_logits[delcpoy_id,:]
                #     after_tem_split = tem[delcpoy_id,:]
                #     gen = m.Bias(after_tem_split[0], L=None, inputdata=after_class_logits_split[0], val=val,
                #                  logger=logger)
                #     class_logits[delcpoy_id, :] = after_class_logits_split[0] + gen
                #
                # else:
                # ##########
                #     after_class_logits_split = []
                #     after_tem_split = []
                #     class_logits_split = torch.chunk(class_logits, 3, dim=0)
                #     tem_split = torch.chunk(tem, 3, dim=0)
                #     for id_tem, cla_tem, ttt in zip(delcpoy_id, class_logits_split,tem_split):
                #         after_class_logits_split.append(cla_tem[id_tem, :])
                #         after_tem_split.append(ttt[id_tem, :, :, :])
                #
                #     for j in range(3):
                #         gen = m.Bias(after_tem_split[j], L=None, inputdata=after_class_logits_split[j],val=val,
                #                      logger=logger)
                #         #after_class_logits_split[j] = after_class_logits_split[j] + gen
                #         class_logits_split[j][delcpoy_id[j],:] = after_class_logits_split[j] + gen
                #     class_logits = torch.cat( class_logits_split, dim=0)
                #################
                
                ################# 11 25
                x, result, save_id = self.post_processor((x, class_logits, box_regression), proposals)
              
                delcpoy_id = save_id 
 
                fliter_box_prop,new_id = self.sel_prop(copy_pro,delcpoy_id)
                delcpoy_id  = new_id
                #####  重新计算
                x1,_= self.feature_extractor(features, fliter_box_prop)
                class_logits_1, box_regression_1 = self.predictor(x1)
                #assert int(sum(class_logits_1[0] == class_logits[int(save_id[0][0])])) == 151
                
                ######
                
                union_id = self.sel_uni_id(fliter_box_prop,logger = logger)
                union_proposals = []
                for proposal, rel_pair_idx in zip(fliter_box_prop, union_id):
                    rel_pair_idx = torch.Tensor(rel_pair_idx).long().cuda()
                    head_proposal = proposal[rel_pair_idx[:, 0]]
                    mid_proposal = proposal[rel_pair_idx[:, 1]]
                    tail_proposal = proposal[rel_pair_idx[:, 2]]
                    # t = 1
                    union_proposal = boxlist_union(head_proposal,tail_proposal,flag=2, boxlist_mid=mid_proposal)
                    union_proposals.append(union_proposal)
           
                _,uni_fea= self.feature_extractor(features, union_proposals)
                gen = m.Bias(G=uni_fea, L=None, inputdata=class_logits_1,val=val,
                                      logger=logger)
                
                if len(result) < 2:
                    print(len(result[0]),len(class_logits_1))
                    xiu1 = class_logits_1 + gen
                    assert xiu1.numel() != 0
     
                    xiu1_socres = F.softmax(xiu1,-1)
                    xiu1_rel_scores, xiu_1_rel_class = xiu1_socres.max(dim=1)
                  
            
                    cpoy_result = copy.deepcopy(result)
                    cpoy_result[0].extra_fields["pred_labels"] = xiu_1_rel_class
                    cpoy_result[0].extra_fields["pred_scores"] =  xiu1_rel_scores
                 
                
                    result = cpoy_result
                    
                    if self.cfg.TEST.SAVE_PROPOSALS:
                        _, sort_ind = result.get_field("pred_scores").view(-1).sort(dim=0, descending=True)
                        x = x[sort_ind]
                        result = result[sort_ind]
                        result.add_field("features", x.cpu().numpy())

                    return x, result, {}
            
                else:
                    print(len(result[0]),len(result[1]),len(class_logits_1))
                # if len(class_logits_1) != 512:
                #       t = 1
                    xiu1 = class_logits_1 + gen
                    assert xiu1.numel() != 0
                    xiu2,xiu3 = xiu1[0:len(result[0])],xiu1[len(result[0]):]
                    xiu2_socres = F.softmax(xiu2,-1)
                    xiu3_socres = F.softmax(xiu3,-1)
                
                    xiu2_rel_scores, xiu_2_rel_class = xiu2_socres.max(dim=1)
                    xiu3_rel_scores, xiu_3_rel_class = xiu3_socres.max(dim=1)
                    
            
                    cpoy_result = copy.deepcopy(result)
                    cpoy_result[0].extra_fields["pred_labels"] = xiu_2_rel_class
                    cpoy_result[0].extra_fields["pred_scores"] =  xiu2_rel_scores
                    cpoy_result[1].extra_fields["pred_labels"] = xiu_3_rel_class
                    cpoy_result[1].extra_fields["pred_scores"] =  xiu3_rel_scores
                
                    result = cpoy_result
                    
                    if self.cfg.TEST.SAVE_PROPOSALS:
                        _, sort_ind = result.get_field("pred_scores").view(-1).sort(dim=0, descending=True)
                        x = x[sort_ind]
                        result = result[sort_ind]
                        result.add_field("features", x.cpu().numpy())

                    return x, result, {}
                
                ###############
            #     x, result, save_id = self.post_processor((x, class_logits, box_regression), proposals)
            #    # assert np.argmax(np.array(torch.softmax(class_logits[int(save_id[0][10])].cpu(),dim=-1))) == int(result[0].extra_fields["pred_labels"][10])




            #     # if we want to save the proposals, we need sort them by confidence first.
            #     if self.cfg.TEST.SAVE_PROPOSALS:
            #         _, sort_ind = result.get_field("pred_scores").view(-1).sort(dim=0, descending=True)
            #         x = x[sort_ind]
            #         result = result[sort_ind]
            #         result.add_field("features", x.cpu().numpy())

            #     return x, result, {}

        else:
            if not self.training:
                x, result, _ = self.post_processor((x, class_logits, box_regression), proposals)

                # if we want to save the proposals, we need sort them by confidence first.
                if self.cfg.TEST.SAVE_PROPOSALS:
                    _, sort_ind = result.get_field("pred_scores").view(-1).sort(dim=0, descending=True)
                    x = x[sort_ind]
                    result = result[sort_ind]
                    result.add_field("features", x.cpu().numpy())

            return x, result, {}




        ####


    # if self.training:
    #     # Faster R-CNN subsamples during training the proposals with a fixed
    #     # positive / negative ratio
    #     with torch.no_grad():
    #         proposals = self.samp_processor.subsample(proposals, targets)
    #
    # # extract features that will be fed to the final classifier. The
    # # feature_extractor generally corresponds to the pooler + heads
    # x = self.feature_extractor(features, proposals)
    # # final classifier that converts the features into predictions
    # class_logits, box_regression = self.predictor(x)
    #
    # #### 117 修改
    # rel_tem = copy.deepcopy(class_logits.detach())
    # label_tem = copy.deepcopy(torch.cat(targets, -1).detach())
    # # gen = m.Bias(S, uni_tem, val=val, logger=logger)
    # loss_mse, gen = m.Bias(x, uni_tem=None, inputdata=rel_tem, iteration=iteration, ff=label_tem, val=val,
    #                        logger=logger)
    #
    # ####
    #
    # if not self.training:
    #     x, result = self.post_processor((x, class_logits, box_regression), proposals)
    #
    #     # if we want to save the proposals, we need sort them by confidence first.
    #     if self.cfg.TEST.SAVE_PROPOSALS:
    #         _, sort_ind = result.get_field("pred_scores").view(-1).sort(dim=0, descending=True)
    #         x = x[sort_ind]
    #         result = result[sort_ind]
    #         result.add_field("features", x.cpu().numpy())
    #
    #     return x, result, {}
    #
    # loss_classifier, loss_box_reg = self.loss_evaluator([class_logits], [box_regression], proposals)
    #
    # return x, proposals, dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)

def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)
