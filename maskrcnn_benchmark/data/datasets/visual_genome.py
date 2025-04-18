import os
import sys
import torch
import h5py
import json
from PIL import Image
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
import mmcv
import cv2
from itertools import chain
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from PIL import Image, ImageDraw, ImageFont
# from maskrcnn_benchmark.config import cfg
from mmdet.datasets.pipelines.loading import LoadImageFromFile
from mmdet.datasets.pipelines.loading import LoadAnnotations
from mmrotate.datasets.pipelines.transforms import RResize
from PIL import ImageFile
import os.path as osp
BOX_SCALE = 2000 #1024  # Scale at which we have the boxes
from mmrotate.core import  poly2obb_np
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
# import config


class VGDataset(torch.utils.data.Dataset):

    def __init__(self, split, img_dir, roidb_file, dict_file, image_file, transforms=None,
                 filter_empty_rels=True, num_im=-1, num_val_im=5000,
                 filter_duplicate_rels=True, filter_non_overlap=True, flip_aug=False, custom_eval=False,
                 custom_path='',cfg = None,mode = None,sta = False):
      
        """
        Torch dataset for VisualGenome
        Parameters:
            split: Must be train, test, or val
            img_dir: folder containing all vg images
            roidb_file:  HDF5 containing the GT boxes, classes, and relationships
            dict_file: JSON Contains mapping of classes/relationships to words
            image_file: HDF5 containing image filenames
            filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
            filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
            num_im: Number of images in the entire dataset. -1 for all images.
            num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        """

        # if cfg.Type != "CV":
        mmcv =  cfg["mmcv"] if (cfg is not None and cfg.Type != "CV" and not sta ) else None 
        if mmcv is not None:

            if "OBB" in cfg.Type  :

                from mmrotate.datasets.custom_test_4_load import CustomDataset 
                self.cus_data = CustomDataset(ann_file=mmcv["ann_file"],pipeline=mmcv["pipeline"],img_prefix= img_dir + "/") # 
                # self.cus_data_sgdet = CustomDataset(ann_file=mmcv["ann_file"],pipeline=mmcv["pipeline"],img_prefix= img_dir + "/") # 
                SGDT_test = [{'type': 'LoadImageFromFile'}, {'type': 'LoadAnnotations', 'with_bbox': True},  {'type': 'RResize', 'img_scale': (1024, 1024)},{'type': 'MultiScaleFlipAug', 'scale_factor': 1.0, 'flip': False, 'transforms': [{'type': 'Normalize', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_rgb': True}, {'type': 'Pad', 'size_divisor': 32}, {'type': 'DefaultFormatBundle'}, {'type': 'Collect', 'keys': ['img', 'gt_bboxes', 'gt_labels']}]}]
                SGDT = [{'type': 'LoadImageFromFile'}, {'type': 'LoadAnnotations', 'with_bbox': True}, {'type': 'RResize', 'img_scale': (1024, 1024)}, {'type': 'Normalize', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_rgb': True}, {'type': 'Pad', 'size_divisor': 32}, {'type': 'DefaultFormatBundle'}, {'type': 'Collect', 'keys': ['img', 'gt_bboxes', 'gt_labels']}]
                self.cus_data_sgdet = CustomDataset(ann_file=mmcv["ann_file"],pipeline=SGDT,img_prefix= img_dir + "/") # 
                self.pipeline_sgdet = self.cus_data_sgdet.pipeline
                
                self.cus_data_sgdet_test = CustomDataset(ann_file=mmcv["ann_file"],pipeline=SGDT_test,img_prefix= img_dir + "/") # 
                self.pipeline_sgdet_test = self.cus_data_sgdet_test.pipeline
                ######
                self.pipeline = self.cus_data.pipeline
                self.pre_pipeline = self.cus_data.pre_pipeline

            elif "HBB" in cfg.Type :

                from mmdetection_RS.mmdet.datasets.custom_RS import CustomDataset_RS_HBB
                self.cus_data = CustomDataset_RS_HBB(ann_file=mmcv["ann_file"],pipeline=mmcv["pipeline"],img_prefix= img_dir + "/") # 
         
                SGDT = [{'type': 'LoadImageFromFile'}, {'type': 'LoadAnnotations', 'with_bbox': True}, {'type': 'Resize', 'img_scale': (1024, 1024), 'keep_ratio': True}, {'type': 'Normalize', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_rgb': True}, {'type': 'Pad', 'size_divisor': 32}, {'type': 'DefaultFormatBundle'}, {'type': 'Collect', 'keys': ['img', 'gt_bboxes', 'gt_labels']}]
                SGDT_test = [{'type': 'LoadImageFromFile'}, {'type': 'LoadAnnotations', 'with_bbox': True}, {'type': 'Resize', 'img_scale': (1024, 1024), 'keep_ratio': True}, {'type': 'MultiScaleFlipAug', 'scale_factor': 1.0, 'flip': False, 'transforms': [{'type': 'Normalize', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_rgb': True}, {'type': 'Pad', 'size_divisor': 32},  {'type': 'DefaultFormatBundle'}, {'type': 'Collect', 'keys': ['img', 'gt_bboxes', 'gt_labels']}]}]               
    
                #  for train
        
                self.cus_data_sgdet = CustomDataset_RS_HBB(ann_file=mmcv["ann_file"],pipeline=SGDT,img_prefix= img_dir + "/") # 
                self.pipeline_sgdet = self.cus_data_sgdet.pipeline

                self.pipeline = self.cus_data.pipeline
                self.cus_data_sgdet_test = CustomDataset_RS_HBB(ann_file=mmcv["ann_file"],pipeline=SGDT_test,img_prefix= img_dir + "/") 
                self.pipeline_sgdet_test = self.cus_data_sgdet_test.pipeline
                self.pre_pipeline = self.cus_data.pre_pipeline



        
        assert split in {'train', 'val', 'test'}
        self.flip_aug =  flip_aug
        self.split = split
        self.img_dir = img_dir
        self.dict_file = dict_file
        self.roidb_file = roidb_file
        self.image_file = image_file
        self.filter_non_overlap = filter_non_overlap and self.split == 'train'
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'
        self.transforms = transforms

        self.ind_to_classes, self.ind_to_predicates, self.ind_to_attributes, self.predicate_to_ind,  self.predicate_count,  self.object_count = load_info(
            dict_file
        )  # contiguous 151, 51 containing __background__
        self.categories = {
            i: self.ind_to_classes[i] for i in range(len(self.ind_to_classes))
        }


               

        self.custom_eval = custom_eval
        if self.custom_eval:
            self.get_custom_imgs(custom_path)
        else:

            self.filenames, self.img_info = load_image_filenames(img_dir, image_file) 
            filenames = self.filenames
            img_info = self.img_info

            
            if cfg is not None: 
                if cfg.Type == "CV"  :  

 
                    
                    self.split_mask, self.gt_boxes, self.gt_classes, self.gt_attributes, self.relationships = load_graphs_CV(
                        self.roidb_file, self.split, num_im, num_val_im=num_val_im,
                        filter_empty_rels=filter_empty_rels,
                        filter_non_overlap=self.filter_non_overlap,
                    )


                else:


                    self.split_mask, self.gt_boxes, self.gt_classes, self.gt_attributes, self.relationships, self.poly, self.four, self.poly_8, self.anglenotle90 = load_graphs(
                        self.roidb_file,
                        self.split, 
                        num_im, 
                        num_val_im=num_val_im,
                        filter_empty_rels=filter_empty_rels,
                        filter_non_overlap=self.filter_non_overlap,
                        filenames = filenames,
                        img_info=img_info
                        )
            
            
                self.filenames = [self.filenames[i] for i in np.where(self.split_mask)[0]]
                self.img_info = [self.img_info[i] for i in np.where(self.split_mask)[0]]
                

            
            if cfg is not None  :

                self.mode = mode
                self.cfg = cfg
                
               
                if cfg.Type != "CV" :
                    if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX and cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                            self.mode = "Predcls"
                    elif cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX and not cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                            self.mode = "Sgcls"
                    elif not cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX and not cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                            self.mode = "Sgdets"
                            
                    
                    self.type = cfg.Type
                    if "OBB" in cfg.Type :
                        #### 构建info
                        self.data_infos = []
                        self.ann_infos = []
                        for kk in range(len(self.filenames)):
                            data_info = {}
                            ann_info = {}
                            
                            data_info['filename'] = os.path.basename(self.filenames[kk])
                            data_info['ann'] = {}

                            ann_info['bboxes'] = self.poly[kk]
                            ann_info['labels'] = self.gt_classes[kk]

                            ann_info['bboxes_ignore'] = np.zeros(
                                (0, 5), dtype=np.float32)
                            ann_info['labels_ignore'] = np.array(
                                [], dtype=np.int64)
                            ann_info['polygons_ignore'] = np.zeros(
                                (0, 8), dtype=np.float32)

                            data_info['ann']['bboxes'] = self.poly[kk]
                            data_info['ann']['labels'] = self.gt_classes[kk]


                            data_info['ann']['bboxes_ignore'] = np.zeros(
                                (0, 5), dtype=np.float32)
                            data_info['ann']['labels_ignore'] = np.array(
                                [], dtype=np.int64)
                            data_info['ann']['polygons_ignore'] = np.zeros(
                                (0, 8), dtype=np.float32)
                            
                            self.data_infos.append(data_info)
                            self.ann_infos.append(ann_info)

                    elif "HBB" in cfg.Type :

                        #### 构建info
                        self.data_infos = []
                        self.ann_infos = []
                        for kk in range(len(self.filenames)):
                            data_info = {}
                            ann_info = {}
                            
                            data_info['filename'] = os.path.basename(self.filenames[kk])
                            data_info['ann'] = {}

                            ann_info['bboxes'] = self.gt_boxes[kk]
                            ann_info['labels'] = self.gt_classes[kk]
                        
                            ann_info['bboxes_ignore'] = np.zeros(
                                (0, 4), dtype=np.float32)
                            ann_info['labels_ignore'] = np.array(
                                [], dtype=np.int64)
                            ann_info['polygons_ignore'] = np.zeros(
                                (0, 4), dtype=np.float32)

                            data_info['ann']['bboxes'] = self.gt_boxes[kk]
                            data_info['ann']['labels'] = self.gt_classes[kk]
        
                            data_info['ann']['bboxes_ignore'] = np.zeros(
                                (0, 4), dtype=np.float32)
                            data_info['ann']['labels_ignore'] = np.array(
                                [], dtype=np.int64)
                            data_info['ann']['polygons_ignore'] = np.zeros(
                                (0, 8), dtype=np.float32)
                            
                            self.data_infos.append(data_info)
                            self.ann_infos.append(ann_info)
                        
                        

      
    def __getitem__(self, index):   ### 按照batch 去迭代

        if self.cfg.Type == "CV" :
            if self.custom_eval:
                img = Image.open(self.custom_files[index]).convert("RGB")
                target = torch.LongTensor([-1])
                if self.transforms is not None:
                    img, target = self.transforms(img, target)
                return img, target, index

            img = Image.open(self.filenames[index]).convert("RGB")
            if img.size[0] != self.img_info[index]['width'] or img.size[1] != self.img_info[index]['height']:
                print('=' * 20, ' ERROR index ', str(index), ' ', str(img.size), ' ', str(self.img_info[index]['width']),
                    ' ', str(self.img_info[index]['height']), ' ', '=' * 20)

            flip_img = (random.random() > 0.5) and self.flip_aug and (self.split == 'train')

            target = self.get_groundtruth(index, flip_img)

            if flip_img:
                img = img.transpose(method=Image.FLIP_LEFT_RIGHT)

            if self.transforms is not None:
                img, target = self.transforms(img, target)

            return img, target, index
        
        else:
            


         
            img_info = self.data_infos[index]
            ann_info = self.ann_infos[index]
            results = dict(img_info=img_info, ann_info=ann_info)
    

            self.pre_pipeline(results)
            results_sgdt = copy.deepcopy(results)

            data = self.pipeline(results)
           
            #### for Sgdet
            if self.split != 'train':

                 img = data["img"][0].data
                 target = self.RS_get_groundtruth(data,index)

                 if self.mode == "Sgdets"  and "Large" in self.type:
                    data1 =  self.pipeline_sgdet_test(results_sgdt)
                    target1 = self.RS_get_groundtruth(data1,index,evaluation = True,flag= True)
                    target.extra_fields["target1"] = target1
                    target.extra_fields["data1"] = data1         

                    return img, target, index, data1["img"].data if not isinstance(data1["img"], list) else data1["img"][0].data, target1
                 return img, target, index  
            
            else:
                
                img = data["img"].data
                target = self.RS_get_groundtruth(data,index)

                if self.mode == "Sgdets" and "Large" in self.type:
                    data1 =  self.pipeline_sgdet(results_sgdt)
                    target1 = self.RS_get_groundtruth(data1,index)
                    target.extra_fields["data1"] = data1
                    target.extra_fields["target1"] = target1
                    
                    return img, target, index, data1["img"].data if not isinstance(data1["img"], list) else data1["img"][0].data, target1

                return img, target, index 




    def get_statistics(self,cfg= None,sta = False):
        fg_matrix, bg_matrix = get_VG_statistics(img_dir=self.img_dir, roidb_file=self.roidb_file,
                                                 dict_file=self.dict_file,
                                                 image_file=self.image_file, must_overlap=True,cfg = cfg,sta = sta)
        eps = 1e-3
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix
        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)

        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'obj_classes': self.ind_to_classes,
            'rel_classes': self.ind_to_predicates,
            'att_classes': self.ind_to_attributes,
        }
        return result

    def get_custom_imgs(self, path):
        self.custom_files = []
        self.img_info = []
        for file_name in os.listdir(path):
            self.custom_files.append(os.path.join(path, file_name))
            img = Image.open(os.path.join(path, file_name)).convert("RGB")
            self.img_info.append({'width': int(img.width), 'height': int(img.height)})

    def get_img_info(self, index):
        # WARNING: original image_file.json has several pictures with false image size
        # use correct function to check the validity before training
        # it will take a while, you only need to do it once

        # correct_img_info(self.img_dir, self.image_file)
        return self.img_info[index]

    def RS_get_groundtruth(self, data, index, evaluation=False, flip_img=False,flag = False):

        if self.split != 'train':

            if "HBB" in self.cfg.Type  :
                 
                
                # if self.mode == "Predcls":
                box = data["gt_bboxes"][0].data
                h,w = data["img"][0].data.shape[1],data["img"][0].data.shape[2]
                assert data["gt_bboxes"][0].data.shape[1] == 4  # 是否包含角度
                target = BoxList(box, (w, h), 'xyxy')  


                
            elif "OBB" in self.cfg.Type :
                 
                 box = data["gt_bboxes"][0].data
                 h,w = data["img"][0].data.shape[1],data["img"][0].data.shape[2]
                 assert data["gt_bboxes"][0].data.shape[1] == 5
                
                 target = BoxList(box, (w, h), 'xywha')
           
        
        
        
        else:

            if isinstance(data["gt_bboxes"], list):
                data["gt_bboxes"] = data["gt_bboxes"][0]
                data["img"] = data["img"][0]
            box = data["gt_bboxes"].data
        #    isinstance(my_variable, list):
            h,w = data["img"].data.shape[1],data["img"].data.shape[2]

            if "HBB" in self.cfg.Type :
                 box = data["gt_bboxes"].data
                 h,w = data["img"].data.shape[1],data["img"].data.shape[2]
                 assert data["gt_bboxes"].data.shape[1] == 4  # 是否包含角度
                 target = BoxList(box, (w, h), 'xyxy')  

            elif "OBB" in self.cfg.Type :
                 box = data["gt_bboxes"].data
                 h,w = data["img"].data.shape[1],data["img"].data.shape[2]
                 assert data["gt_bboxes"].data.shape[1] == 5

                 target = BoxList(box, (w, h), 'xywha')
        

        target.add_field("labels", torch.from_numpy(self.gt_classes[index]))
        # target.add_field("attributes", torch.from_numpy(self.gt_attributes[index]))
        target.add_field("data", data)

        relation = self.relationships[index].copy()  # (num_rel, 3)

        

        # add relation to target
        num_box = len(target)
        relation_map = torch.zeros((num_box, num_box), dtype=torch.int64)
        for i in range(relation.shape[0]):
            if relation_map[int(relation[i, 0]), int(relation[i, 1])] > 0:
                if (random.random() > 0.5):
                    relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
            else:
                relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
        target.add_field("relation", relation_map, is_triplet=True)



        if evaluation:
     
            target.add_field("relation_tuple", torch.LongTensor(relation))  # for evaluation

        
        return target
    
    def RS_test_get_groundtruth(self, index, evaluation=False, flip_img=False):
        img_info = self.get_img_info(index) # {'width': 500, 'url': 'https://cs.stanford.edu/people/rak248/VG_100K/2355667.jpg', 'height': 400, 'image_id': 2355667, 'coco_id': None, 'flickr_id': 538622274, 'anti_prop': 0.03171130260801489}
        w, h = img_info['width'], img_info['height']  ## {'image_id': 421, 'height': 8633, 'width': 5719}
        # important: recover original box from BOX_SCALE
        box = self.ann_infos[index]['bboxes']
        w_f = 1024/w
        h_f = 1024/h
       
        if box.shape[-1] == 5:
            box[:,0] *= w_f
            box[:,1] *= h_f
            box[:,2:4] *= np.sqrt(w_f*h_f)
            ### 缩放
            assert box.shape[-1] == 5  # guard against no boxes
            target = BoxList(box, (w, h), 'xywha')
        else:
            box[:,0] *= w_f
            box[:,1] *= h_f
            box[:,2] *= w_f
            box[:,3] *= h_f
            ### 缩放
            assert box.shape[-1] == 4  # guard against no boxes
            target = BoxList(box, (w, h), 'xyxy')          

        target.add_field("labels", torch.from_numpy(self.gt_classes[index]))
        target.add_field("attributes", torch.from_numpy(self.gt_attributes[index]))

        relation = self.relationships[index].copy()  # (num_rel, 3)
        if self.filter_duplicate_rels:  # 过滤重复条目
            # Filter out dupes!
            assert self.split == 'train'
            old_size = relation.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in relation:
                all_rel_sets[(o0, o1)].append(r)
            relation = [(k[0], k[1], np.random.choice(v)) for k, v in all_rel_sets.items()]
            relation = np.array(relation, dtype=np.int32)

        # add relation to target
        num_box = len(target)
        relation_map = torch.zeros((num_box, num_box), dtype=torch.int64)
        for i in range(relation.shape[0]):
            if relation_map[int(relation[i, 0]), int(relation[i, 1])] > 0:
                if (random.random() > 0.5):
                    relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
            else:
                relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
        target.add_field("relation", relation_map, is_triplet=True)

        target.add_field("relation_tuple", torch.LongTensor(relation))  # for evaluation


        return target
        
    def get_groundtruth(self, index, evaluation=False, flip_img=False):
        img_info = self.get_img_info(index) # {'width': 500, 'url': 'https://cs.stanford.edu/people/rak248/VG_100K/2355667.jpg', 'height': 400, 'image_id': 2355667, 'coco_id': None, 'flickr_id': 538622274, 'anti_prop': 0.03171130260801489}
        w, h = img_info['width'], img_info['height']  ## {'image_id': 421, 'height': 8633, 'width': 5719}
        # important: recover original box from BOX_SCALE
        box = self.gt_boxes[index] / BOX_SCALE * max(w, h) ### 恢复出原始框
        box = torch.from_numpy(box).reshape(-1, 4)  # guard against no boxes
        if flip_img:
            new_xmin = w - box[:, 2]
            new_xmax = w - box[:, 0]
            box[:, 0] = new_xmin
            box[:, 2] = new_xmax
        target = BoxList(box, (w, h), 'xyxy')  # xyxy

        target.add_field("labels", torch.from_numpy(self.gt_classes[index]))
        target.add_field("attributes", torch.from_numpy(self.gt_attributes[index]))

        relation = self.relationships[index].copy()  # (num_rel, 3)
        if self.filter_duplicate_rels:  # 过滤重复条目
            # Filter out dupes!
            assert self.split == 'train'
            old_size = relation.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in relation:
                all_rel_sets[(o0, o1)].append(r)
            relation = [(k[0], k[1], np.random.choice(v)) for k, v in all_rel_sets.items()]
            relation = np.array(relation, dtype=np.int32)

        # add relation to target
        num_box = len(target)
        relation_map = torch.zeros((num_box, num_box), dtype=torch.int64)
        for i in range(relation.shape[0]):
            if relation_map[int(relation[i, 0]), int(relation[i, 1])] > 0:
                if (random.random() > 0.5):
                    relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
            else:
                relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
        target.add_field("relation", relation_map, is_triplet=True)

        if evaluation:
            target = target.clip_to_image(remove_empty=False)
            target.add_field("relation_tuple", torch.LongTensor(relation))  # for evaluation
            return target
        else:
            target = target.clip_to_image(remove_empty=True)
            return target

    def __len__(self):
        if self.custom_eval:
            return len(self.custom_files)
        return len(self.filenames)


def get_VG_statistics(img_dir, roidb_file, dict_file, image_file, must_overlap=True,cfg = None,sta = False):
    train_data = VGDataset(split='train', img_dir=img_dir, roidb_file=roidb_file,
                           dict_file=dict_file, image_file=image_file, num_val_im=5000,
                           filter_duplicate_rels=False,cfg = cfg,sta = sta, mode='statistic')
    num_obj_classes = len(train_data.ind_to_classes)
    num_rel_classes = len(train_data.ind_to_predicates)
    fg_matrix = np.zeros((num_obj_classes, num_obj_classes, num_rel_classes), dtype=np.int64)
    bg_matrix = np.zeros((num_obj_classes, num_obj_classes), dtype=np.int64)

    for ex_ind in tqdm(range(len(train_data))):
        gt_classes = train_data.gt_classes[ex_ind].copy()
        gt_relations = train_data.relationships[ex_ind].copy()
        gt_boxes = train_data.gt_boxes[ex_ind].copy()

        # For the foreground, we'll just look at everything
        o1o2 = gt_classes[gt_relations[:, :2]]
        for (o1, o2), gtr in zip(o1o2, gt_relations[:, 2]):
            fg_matrix[o1, o2, gtr] += 1
        # For the background, get all of the things that overlap.
        o1o2_total = gt_classes[np.array(
            box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
        for (o1, o2) in o1o2_total:
            bg_matrix[o1, o2] += 1

    return fg_matrix, bg_matrix


def box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations. 
    If no overlapping boxes, use all of them."""
    n_cands = boxes.shape[0]

    overlaps = bbox_overlaps(boxes.astype(np.float), boxes.astype(np.float), to_move=0) > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=np.bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))

        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes


def bbox_overlaps(boxes1, boxes2, to_move=1):
    """
    boxes1 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    boxes2 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    """
    # print('boxes1: ', boxes1.shape)
    # print('boxes2: ', boxes2.shape)
    num_box1 = boxes1.shape[0]
    num_box2 = boxes2.shape[0]
    lt = np.maximum(boxes1.reshape([num_box1, 1, -1])[:, :, :2], boxes2.reshape([1, num_box2, -1])[:, :, :2])  # [N,M,2]
    rb = np.minimum(boxes1.reshape([num_box1, 1, -1])[:, :, 2:], boxes2.reshape([1, num_box2, -1])[:, :, 2:])  # [N,M,2]

    wh = (rb - lt + to_move).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return inter


def correct_img_info(img_dir, image_file):
    with open(image_file, 'r') as f:
        data = json.load(f)
    for i in range(len(data)):
        img = data[i]
        basename = '{}.jpg'.format(img['image_id'])
        filename = os.path.join(img_dir, basename)
        img_data = Image.open(filename).convert("RGB")
        if img['width'] != img_data.size[0] or img['height'] != img_data.size[1]:
            print('--------- False id: ', i, '---------')
            print(img_data.size)
            print(img)
            data[i]['width'] = img_data.size[0]
            data[i]['height'] = img_data.size[1]
    with open(image_file, 'w') as outfile:
        json.dump(data, outfile)


def load_info(dict_file, add_bg=True):
    """
    Loads the file containing the visual genome label meanings
    """
    info = json.load(open(dict_file, 'r'))
    if add_bg:
        info['label_to_idx']['__background__'] = 0
        info['predicate_to_idx']['__background__'] = 0
        info['attribute_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_idx']
    predicate_count = info['predicate_count']
    predicate_to_ind = info['predicate_to_idx']
    object_count = info['object_count']
    attribute_to_ind = info['attribute_to_idx']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])
    ind_to_attributes = sorted(attribute_to_ind, key=lambda k: attribute_to_ind[k])

    #return ind_to_classes, ind_to_predicates, ind_to_attributes
    return ind_to_classes, ind_to_predicates, ind_to_attributes, predicate_to_ind, predicate_count,object_count

def load_image_filenames_CV(img_dir, image_file):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    Parameters:
        image_file: JSON file. Elements contain the param "image_id".
        img_dir: directory where the VisualGenome images are located
    Return: 
        List of filenames corresponding to the good images
    """
    with open(image_file, 'r') as f:
        im_data = json.load(f)

    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    fns = []
    img_info = []
    for i, img in enumerate(im_data):
        basename = '{}.jpg'.format(img['image_id'])
        if basename in corrupted_ims:
            continue

        filename = os.path.join(img_dir, basename)
        if os.path.exists(filename):
            fns.append(filename)
            img_info.append(img)
    assert len(fns) == 108073
    assert len(img_info) == 108073
    return fns, img_info


def load_image_filenames(img_dir, image_file):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    Parameters:
        image_file: JSON file. Elements contain the param "image_id".
        img_dir: directory where the VisualGenome images are located
    Return:
        List of filenames corresponding to the good images
    """



    with open(image_file, "r") as f:
        im_data = json.load(f)
    

    fns = []
    img_info = []
    for i, img in enumerate(im_data):

        basename = str(img["image_id"]).zfill(4) + ".png"

        filename = os.path.join(img_dir, basename)

        if os.path.exists(filename):
            fns.append(filename)
            img_info.append(img)


    return fns, img_info


def load_graphs(roidb_file, split, num_im, num_val_im, filter_empty_rels, filter_non_overlap, train_index=None,
                val_index=None, test_index=None,filenames = None, img_info = None):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    Parameters:
        roidb_file: HDF5
        split: (train, val, or test)
        num_im: Number of images we want
        num_val_im: Number of validation images
        filter_empty_rels: (will be filtered otherwise.)
        filter_non_overlap: If training, filter images that dont overlap.
    Return: 
        image_index: numpy array corresponding to the index of images we're using
        boxes: List where each element is a [num_gt, 4] array of ground 
                    truth boxes (x1, y1, x2, y2)
        gt_classes: List where each element is a [num_gt] array of classes
        relationships: List where each element is a [num_r, 3] array of 
                    (box_ind_1, box_ind_2, predicate) relationships
    """

    train_index = [352, 1140, 781, 734, 138, 942, 319, 947, 461, 1187, 783, 714, 1078, 1117, 1193, 1047, 1113, 1108, 956, 1136, 987, 948, 560, 295, 835, 458, 444, 152, 326, 1016, 843, 278, 269, 462, 474, 137, 1037, 874, 270, 968, 1185, 503, 694, 997, 297, 616, 219, 387, 1195, 1197, 161, 840, 99, 713, 607, 431, 1036, 810, 403, 672, 992, 1133, 553, 1231, 634, 969, 554, 1121, 97, 93, 402, 1045, 1202, 925, 436, 375, 580, 282, 215, 768, 496, 647, 96, 472, 411, 735, 928, 718, 750, 746, 1056, 250, 1219, 1039, 902, 351, 1035, 1046, 1034, 490, 155, 234, 965, 88, 299, 678, 98, 487, 876, 192, 1033, 1073, 1253, 418, 1272, 144, 1070, 497, 499, 638, 575, 1040, 291, 854, 578, 1223, 1042, 917, 954, 376, 410, 257, 769, 432, 87, 752, 1218, 821, 799, 633, 686, 24, 89, 43, 434, 271, 1051, 1166, 571, 708, 715, 405, 908, 124, 583, 635, 48, 782, 1190, 1067, 511, 423, 59, 829, 324, 1247, 421, 627, 381, 858, 1260, 364, 559, 368, 941, 657, 316, 313, 33, 816, 170, 393, 332, 813, 915, 1212, 913, 592, 819, 609, 494, 54, 101, 919, 255, 848, 812, 171, 356, 1097, 443, 424, 60, 665, 1012, 391, 757, 709, 169, 762, 493, 955, 727, 1142, 429, 654, 624, 153, 392, 446, 615, 427, 1096, 515, 775, 682, 996, 388, 576, 287, 286, 83, 629, 832, 646, 979, 1205, 881, 378, 1264, 645, 516, 106, 359, 1132, 262, 1144, 475, 274, 1244, 339, 435, 618, 1220, 1111, 145, 1227, 1221, 42, 846, 766, 568, 641, 921, 814, 304, 864, 483, 1110, 442, 1116, 412, 1158, 437, 139, 510, 147, 1058, 450, 1269, 1137, 249, 127, 310, 828, 801, 690, 972, 1204, 826, 946, 1165, 166, 25, 92, 983, 502, 1225, 1168, 1206, 414, 300, 619, 631, 336, 1189, 1245, 441, 251, 877, 1094, 770, 151, 401, 974, 140, 1064, 808, 459, 964, 834, 1248, 522, 1055, 374, 679, 361, 23, 796, 1228, 594, 579, 512, 79, 659, 253, 358, 85, 649, 221, 1242, 628, 484, 1072, 588, 593, 1109, 922, 1145, 1066, 537, 501, 818, 860, 802, 107, 273, 765, 971, 457, 491, 100, 447, 863, 1182, 1211, 382, 321, 705, 544, 266, 505, 980, 1159, 1170, 985, 489, 284, 1075, 1048, 850, 867, 891, 482, 428, 967, 47, 728, 178, 582, 958, 1240, 261, 685, 349, 1243, 780, 803, 732, 479, 900, 150, 181, 892, 384, 202, 477, 856, 355, 298, 22, 1246, 385, 725, 492, 513, 1217, 70, 1014, 82, 148, 149, 301, 55, 1157, 622, 620, 506, 1258, 1038, 551, 1041, 1149, 338, 589, 37, 471, 561, 1059, 476, 790, 625, 320, 478, 509, 792, 500, 683, 702, 448, 644, 331, 240, 590, 556, 426, 1201, 1259, 495, 102, 674, 488, 377, 1226, 789, 383, 562, 211, 1122, 445, 1252, 263, 861, 651, 610, 105, 227, 658, 570, 422, 1169, 699, 1148, 878, 668, 289, 49, 1135, 259, 416, 630, 951, 275, 684, 930, 182, 572, 1254, 62, 653, 817, 110, 774, 517, 1203, 538, 940, 130, 470, 587, 640, 128, 1, 1210, 953, 1120, 648, 90, 94, 1134, 1268, 486, 1208, 667, 932, 317, 1229, 11, 541, 777, 784, 844, 146, 330, 420, 567, 542, 737, 585, 529, 1052, 1054, 51, 1261, 173, 1118, 1126, 113, 16, 439, 347, 81, 404, 91, 507, 927, 343, 973, 120, 123, 759, 413, 76, 833, 315, 409, 322, 72, 1119, 1129, 700, 1266, 329, 912, 1114, 1237, 348, 239, 1050, 772, 109, 379, 574, 417, 531, 931, 357, 652, 1071, 1090, 1125, 210, 1060, 692, 256, 719, 943, 710, 581, 438, 342, 696, 296, 1178, 354, 345, 905, 566, 981, 125, 473, 952, 469, 1092, 84, 386, 337, 389, 724, 154, 632, 669, 508, 498, 254, 309, 362, 344, 577, 754, 1251, 637, 617, 1057, 880, 1107, 1061, 1171, 540, 57, 131, 643, 485, 806, 736, 1156, 430, 134, 1230, 639, 1241, 19, 408, 614, 807, 966, 906, 290, 1130, 277, 247, 180, 1235, 68, 548, 1131, 103, 1224, 743, 1167, 642, 1139, 1053, 1160, 655, 467, 698, 831, 779, 1063, 78, 584, 226, 693, 899, 129, 175, 135, 680, 785, 611, 933, 449, 872, 168, 1128, 1153, 328, 1270, 236, 1141, 1164, 75, 285, 546, 764, 468, 306, 303, 220, 260, 1267, 367, 712, 751, 390, 547, 56, 901, 504, 841, 114, 363, 518, 636, 159, 142, 761, 0, 1049, 563, 419, 156, 1216, 804, 1249, 519, 258, 433, 1127, 455, 883, 1115, 44, 1222, 373, 706, 311, 279, 703, 845, 463, 586, 608, 118, 1262, 350, 293, 172, 903, 460, 30, 849, 415, 934, 963, 371, 1043, 552, 957, 1079]
    val_index = [988, 112, 536, 986, 753, 1236, 888, 1154, 778, 189, 80, 936, 938, 1024, 1004, 453, 852, 1020, 28, 360, 1271, 1028, 1076, 1184, 213, 199, 208, 866, 1026, 998, 121, 238, 122, 741, 923, 1022, 664, 822, 1031, 71, 935, 160, 787, 7, 1083, 887, 526, 399, 53, 742, 929, 13, 612, 875, 1069, 697, 837, 176, 95, 786, 896, 191, 241, 217, 820, 885, 886, 6, 1032, 716, 1018, 1186, 598, 452, 163, 909, 1102, 1013, 17, 407, 882, 396, 824, 1214, 722, 924, 206, 74, 1001, 39, 179, 198, 1030, 993, 400, 1234, 398, 29, 1023, 676, 926, 982, 984, 521, 195, 523, 893, 675, 890, 707, 999, 691, 1263, 656, 465, 1029, 663, 1010, 1233, 193, 990, 748, 747, 1257, 731, 805, 671, 242, 451, 8, 721, 111, 38, 894, 895, 1005, 791, 904, 1196, 46, 524, 1172, 897, 970, 14, 1065, 454, 481, 1000, 907, 1199, 898, 879, 825, 1011, 794, 2, 1008, 687, 717, 525, 869, 165, 464, 212, 218, 194, 800, 528, 1003, 797, 1025, 1007, 944, 1093, 744, 599, 224, 3, 720, 1021, 200, 233, 995, 1146, 26, 1006, 726, 606, 204, 466, 244, 1019, 677, 937, 994, 9, 1027, 851, 763, 1124, 406, 873, 771, 600, 1088, 246, 1192, 830, 662, 859, 520, 157, 758, 216, 836, 397, 870, 63, 1207, 58, 425, 776, 939, 661, 395, 695, 650, 749, 1002, 119, 1209, 989, 605, 1017, 1015, 1188, 823, 185, 1238, 738, 527, 1084, 733, 50]
    test_index = [4, 5 , 10, 12, 15, 18, 20, 21, 27, 31, 32, 34, 35, 36, 40, 41, 45, 52, 61, 64, 65, 66, 67, 69, 73, 77, 86, 104, 108, 115, 116, 117, 126, 132, 133, 136, 141, 143, 158, 162, 164, 167, 174, 177, 183, 184, 186, 187, 188, 190, 196, 197, 201, 203, 205, 207, 209, 214, 222, 223, 225, 228, 229, 230, 231, 232, 235, 237, 243, 245, 248, 252, 264, 265, 267, 268, 272, 276, 280, 281, 283, 288, 292, 294, 302, 305, 307, 308, 312, 314, 318, 323, 325, 327, 333, 334, 335, 340, 341, 346, 353, 365, 366, 369, 370, 372, 380, 394, 440, 456, 480, 514, 530, 532, 534, 535, 539, 543, 545, 549, 550, 555, 557, 558, 564, 565, 569, 573, 591, 595, 596, 597, 601, 602, 603, 604, 613, 621, 623, 626, 660, 662, 666, 670, 673, 676, 681, 688, 689, 701, 704, 711, 717, 723, 729, 730, 739, 740, 745, 748, 755, 756, 760, 767, 773, 788, 793, 795, 798, 809, 811, 815, 823, 824, 825, 827, 837, 838, 839, 842, 847, 853, 855, 857, 862, 865, 868, 871, 873, 910, 911, 914, 916, 918, 920, 945, 949, 950, 959, 960, 961, 962, 975, 976, 977, 978, 989, 991, 1009, 1017, 1044, 1062, 1068, 1074, 1077, 1080, 1081, 1082, 1085, 1086, 1087, 1089, 1091, 1095, 1098, 1099, 1100, 1101, 1103, 1104, 1105, 1106, 1112, 1123, 1138, 1143, 1147, 1150, 1151, 1152, 1155, 1161, 1162, 1163, 1173, 1174, 1175, 1176, 1177, 1179, 1180, 1181, 1183, 1191, 1194, 1200, 1213, 1215, 1232, 1239, 1250, 1255, 1256, 1265]
    
    print("Train, Val, Test: ", len(train_index), len(val_index), len(test_index))

    roi_h5 = h5py.File(roidb_file, 'r')
    data_split = roi_h5['split'][:]
    split_mask = np.zeros_like(data_split).astype(bool)

    image_index = np.where(split_mask)[0]
    if split == "train":
        split_mask[train_index] = True
        image_index = np.where(split_mask)[0]
    elif split == "val":
        split_mask[val_index] = True
        image_index = np.where(split_mask)[0]
    else:
        split_mask[test_index] = True    
        image_index = np.where(test_index)[0]


    # Get box information
    all_labels = roi_h5['labels'][:, 0]
    all_attributes = roi_h5['attributes'][:, :]
    all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # cx,cy,w,h

    all_ori = roi_h5['segmentation_{}'.format(BOX_SCALE)][:]  ### 外接矩形

    #### all keys
    ...
    ['active_object_mask', 'attributes', 'boxes_1000', 'boxes_2000', 'img_to_first_box', 
     'img_to_first_rel', 'img_to_first_seg', 'img_to_last_box', 'img_to_last_rel',
       'img_to_last_seg', 'labels', 'predicates', 'relationships', 'segmentation_1000', 'segmentation_2000', 'split']
    ...

    ####
    assert np.all(all_boxes[:, :2] >= 0)  # sanity check
    ###
    for i, box in enumerate(all_boxes):
        if np.any(box[2:] <= 0):
            print(f"Box at index {i} has non-positive width or height: {box}")
            box[2:] = np.maximum(box[2:], 1) 
    ####
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box

    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]
    im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
    im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

    # load relation labels
    _relations = roi_h5['relationships'][:]
    _relation_predicates = roi_h5['predicates'][:, 0]
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

    # Get everything by image.
    boxes = []
    gt_classes = []
    gt_attributes = []
    relationships = []
    poly = []
    for i in range(len(image_index)):
        i_obj_start = im_to_first_box[i]
        i_obj_end = im_to_last_box[i]
        i_rel_start = im_to_first_rel[i]
        i_rel_end = im_to_last_rel[i]

        boxes_i = all_boxes[i_obj_start: i_obj_end + 1, :]
        poly_i = all_ori[i_obj_start: i_obj_end + 1, :]
                         
        gt_classes_i = all_labels[i_obj_start: i_obj_end + 1]
        gt_attributes_i = all_attributes[i_obj_start: i_obj_end + 1, :]

        if i_rel_start >= 0:
            predicates = _relation_predicates[i_rel_start: i_rel_end + 1]
            obj_idx = _relations[i_rel_start: i_rel_end + 1] - i_obj_start  # range is [0, num_box)
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            rels = np.column_stack((obj_idx, predicates))  # (num_rel, 3), representing sub, obj, and pred
        else:
            assert not filter_empty_rels
            rels = np.zeros((0, 3), dtype=np.int32)


        poly.append(poly_i)
        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        gt_attributes.append(gt_attributes_i)
        relationships.append(rels)
    
    #### 恢复出原始
        
    for id in range(len(image_index)):
        img_in = img_info[image_index[id]] 
        w, h = img_in['width'], img_in['height'] 
        boxes[id] =  boxes[id] / BOX_SCALE * 6000
        poly[id] = poly[id] / BOX_SCALE * 6000




    ####     
    rotate = []
    four_point = []
    r_test = []
    assert len(poly) == len(boxes)
    for j in range(len(poly)):
        # rotate.append([])
        li = []
        f = []
        test = []

        for k in range(len(poly[j])):
            p1 = np.float32(poly[j][k][0])
            rect = poly2obb_np(p1, "le90")
            x, y, w, h, a = rect
            if w == 0:
                w = w + 1
            if h == 0:
                h = h + 1
            list_rect = [x, y, w, h, a] 

            ### 保存原始
            p2 = p1.reshape((4, 2))
            p2_rbbox = cv2.minAreaRect(p2)
            rect_test = [p2_rbbox[0][0], p2_rbbox[0][1], p2_rbbox[1][0], p2_rbbox[1][1], p2_rbbox[2]]


           

            rbox = cv2.boxPoints(((x, y), (w, h), a)) 
            rox_list= rbox.tolist()
            flat_list = list(chain(*rox_list))
            f.append(flat_list)
            li.append(list_rect)

            test.append(rect_test)

        
        rotate.append(np.array(li, dtype=np.float32))
        four_point.append(np.array(f,dtype=np.float32))
        r_test.append(np.array(test, dtype=np.float32)) # 保存的原始为经le90的



    #### 去除多余列表
    ploy_8= []
    for tem in poly:
        tem = tem.tolist()
        tem1 = [va[0] for va in tem]
        tem2 = np.array(tem1, dtype=np.float32)
        ploy_8.append(tem2)



    return split_mask, boxes, gt_classes, gt_attributes, relationships, rotate, four_point,ploy_8,r_test



def load_graphs_CV(roidb_file, split, num_im, num_val_im, filter_empty_rels, filter_non_overlap):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    Parameters:
        roidb_file: HDF5
        split: (train, val, or test)
        num_im: Number of images we want
        num_val_im: Number of validation images
        filter_empty_rels: (will be filtered otherwise.)
        filter_non_overlap: If training, filter images that dont overlap.
    Return: 
        image_index: numpy array corresponding to the index of images we're using
        boxes: List where each element is a [num_gt, 4] array of ground 
                    truth boxes (x1, y1, x2, y2)
        gt_classes: List where each element is a [num_gt] array of classes
        relationships: List where each element is a [num_r, 3] array of 
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    roi_h5 = h5py.File(roidb_file, 'r')
    data_split = roi_h5['split'][:]
    split_flag = 2 if split == 'test' else 0
    split_mask = data_split == split_flag

    # Filter out images without bounding boxes
    split_mask &= roi_h5['img_to_first_box'][:] >= 0
    if filter_empty_rels:
        split_mask &= roi_h5['img_to_first_rel'][:] >= 0

    image_index = np.where(split_mask)[0]
    if num_im > -1:
        image_index = image_index[:num_im]
    if num_val_im > 0:
        if split == 'val':
            image_index = image_index[:num_val_im]
        elif split == 'train':
            image_index = image_index[num_val_im:]

    split_mask = np.zeros_like(data_split).astype(bool)
    split_mask[image_index] = True

    # Get box information
    all_labels = roi_h5['labels'][:, 0]
    all_attributes = roi_h5['attributes'][:, :]
    all_boxes = roi_h5['boxes_{}'.format(1024)][:]  # cx,cy,w,h
    assert np.all(all_boxes[:, :2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box

    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]
    im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
    im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

    # load relation labels
    _relations = roi_h5['relationships'][:]
    _relation_predicates = roi_h5['predicates'][:, 0]
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

    # Get everything by image.
    boxes = []
    gt_classes = []
    gt_attributes = []
    relationships = []
    for i in range(len(image_index)):
        i_obj_start = im_to_first_box[i]
        i_obj_end = im_to_last_box[i]
        i_rel_start = im_to_first_rel[i]
        i_rel_end = im_to_last_rel[i]

        boxes_i = all_boxes[i_obj_start: i_obj_end + 1, :]
        gt_classes_i = all_labels[i_obj_start: i_obj_end + 1]
        gt_attributes_i = all_attributes[i_obj_start: i_obj_end + 1, :]

        if i_rel_start >= 0:
            predicates = _relation_predicates[i_rel_start: i_rel_end + 1]
            obj_idx = _relations[i_rel_start: i_rel_end + 1] - i_obj_start  # range is [0, num_box)
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            rels = np.column_stack((obj_idx, predicates))  # (num_rel, 3), representing sub, obj, and pred
        else:
            assert not filter_empty_rels
            rels = np.zeros((0, 3), dtype=np.int32)

        if filter_non_overlap:
            assert split == 'train'
            # construct BoxList object to apply boxlist_iou method
            # give a useless (height=0, width=0)
            boxes_i_obj = BoxList(boxes_i, (1000, 1000), 'xyxy')
            inters = boxlist_iou(boxes_i_obj, boxes_i_obj)
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np.where(rel_overs > 0.0)[0]

            if inc.size > 0:
                rels = rels[inc]
            else:
                split_mask[image_index[i]] = 0
                continue

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        gt_attributes.append(gt_attributes_i)
        relationships.append(rels)

    return split_mask, boxes, gt_classes, gt_attributes, relationships

