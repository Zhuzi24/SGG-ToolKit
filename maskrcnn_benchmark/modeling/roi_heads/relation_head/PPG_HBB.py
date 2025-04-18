from torch.autograd import Variable
import argparse
import copy
import os
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch
from maskrcnn_benchmark.modeling.roi_heads.relation_head.Autoencodermodel import Autoencoder1, Autoencoder2
import random
import shapely
from shapely.geometry import Polygon, Point
import math
from mmdet.core.bbox.iou_calculators import bbox_overlaps
import cv2


class PPG_HBB(nn.Module):
    def __init__(self):
        super(PPG_HBB, self).__init__()
        self.bbox_overlaps = bbox_overlaps

        self.model1 = Autoencoder1(103, 25, 50, 50)
        self.model2 = Autoencoder2(103, 25, 50, 50)

        self.model1 = self.model1.cuda()
        self.model2 = self.model2.cuda()


        current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取脚本所在目录
        model_path = os.path.join(current_dir,'best_model_HBB.pth')

        self.model1.load_state_dict(torch.load(model_path)['model_state_dict1'])
        self.model2.load_state_dict(torch.load(model_path)['model_state_dict2'])
        self.model1.eval()
        self.model2.eval()
        self.criterion = nn.MSELoss(reduction='none')
        



    def calculate_minimum_bounding_box_area(self,boxes):
        # Assuming boxes is a 2D numpy array, where each row is a box
        # represented as [x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7]

        # Find the minimum and maximum x and y coordinates
        x_coords = boxes[:, ::2]  # Get all x coordinates
        y_coords = boxes[:, 1::2]  # Get all y coordinates

        min_x = torch.min(x_coords, axis=1).values
        max_x = torch.max(x_coords, axis=1).values
        min_y = torch.min(y_coords, axis=1).values
        max_y = torch.max(y_coords, axis=1).values

        # The area of the minimum bounding rectangle is (max_x - min_x) * (max_y - min_y)
        areas = (max_x - min_x) * (max_y - min_y)
        
        return areas


    
    def convert_rect_to_points(self,rects):

        x1s = rects[:, 0]
        y1s = rects[:, 1]
        x2s = rects[:, 2]
        y2s = rects[:, 3]
        
        # 创建点坐标数组
        points = torch.stack([x1s, y1s, x2s, y1s, x2s, y2s, x1s, y2s], axis=-1)
        
        return points




    def calculate_HBB_area(self,boxes):
        x1s = boxes[:, 0]
        y1s = boxes[:, 1]
        x2s = boxes[:, 2]
        y2s = boxes[:, 3]

        areas = (x2s - x1s) * (y2s - y1s)

        return areas
    
    def calculate_diagonal_from_rectangle(self,p1):
        # Assuming p1 is a 2D tensor, where each row is a box
        # represented as [x0, y0, x1, y1, x2, y2, x3, y3]

        # Calculate the diagonals
        d1 = torch.sqrt((p1[:, 4] - p1[:, 0])**2 + (p1[:, 5] - p1[:, 1])**2)  
        d2 = torch.sqrt((p1[:, 6] - p1[:, 2])**2 + (p1[:, 7] - p1[:, 3])**2)
        d3 = torch.sqrt((p1[:, 2] - p1[:, 0])**2 + (p1[:, 3] - p1[:, 1])**2)

        # Return the maximum diagonal
        return torch.max(torch.max(d1, d2), d3)

    def calculate_HBB(self, ps1, img_size,head_boxes,tail_boxes):


        def one_hot_encode_batch(num, alphabet_size):
            # cls: (batch_size,) array or list
            one_hot = np.eye(alphabet_size)[num]
            return one_hot
   
         
        spatial_cls = []
        w= img_size[0]
        h= img_size[1]

        row = ps1  # poly + poly + label + label 18

        poly1 = ps1[:,:8]
        poly2 = ps1[:,8:16]

        p = torch.stack((poly1,poly2), axis=-1)
        p = p.reshape(p.shape[0],p.shape[1]*p.shape[2])
        # 找到每对行的最小值和最大值
        pxmin = torch.min(p[:, ::2], axis=-1).values
        pymin = torch.min(p[:, 1::2], axis=-1).values
        pxmax = torch.max(p[:, ::2], axis=-1).values
        pymax = torch.max(p[:, 1::2], axis=-1).values

        uarea=(pxmax-pxmin)*(pymax-pymin)

        ious = torch.tensor(self.bbox_overlaps(  # 533,1003
        head_boxes.float().cuda(),
        tail_boxes.float(),is_aligned=True).cpu().numpy()).cuda()


        area1,area2 = self.calculate_HBB_area(head_boxes), self.calculate_HBB_area(tail_boxes)

        uarea = self.calculate_minimum_bounding_box_area(ps1[:,:16])

        # 计算条件判断的mask
        uarea_zero_mask = (uarea == 0)
        area2_nonzero_mask = (area2 != 0)

        # 用条件判断的mask计算area11和area22
        area11 = torch.where(uarea_zero_mask, area1, area1 / uarea)
        area22 = torch.where(uarea_zero_mask, area2, area2 / uarea)

        # 用条件判断的mask计算areaZ
        areaZ = torch.where(area2_nonzero_mask, area1 / area2, area1)


        # if uarea == 0:
           
        #     area11=area1
        #     area22=area2           
        # else:
        #     area11=area1/uarea
        #     area22=area2/uarea
        
        # if area2 != 0:
        #     areaZ=area1/area2
        # else:
        #     areaZ=area1

        cx1 = torch.mean(poly1[:, ::2], dim=1)  # Get all x coordinates of p1 and calculate mean
        cy1 = torch.mean(poly1[:, 1::2], dim=1)  # Get all y coordinates of p1 and calculate mean
        cx2 = torch.mean(poly2[:, ::2], dim=1)  # Get all x coordinates of p2 and calculate mean
        cy2 = torch.mean(poly2[:, 1::2], dim=1)  # Get all y coordinates of p2 and calculate mean

        # Calculate the distances between the centers
        dist = torch.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
        distimg = math.sqrt(h**2 + w**2)
        distI = dist / distimg

        diagonal1 = self.calculate_diagonal_from_rectangle(poly1)
        diagonal2 = self.calculate_diagonal_from_rectangle(poly2) 

        distZ = torch.zeros_like(diagonal2)
        dist2 = torch.zeros_like(diagonal2)
        dist1 = torch.zeros_like(diagonal1)

        # Calculate distZ and dist2 where diagonal2 is not zero
        mask = diagonal2 != 0
        distZ[mask] = diagonal1[mask] / diagonal2[mask]
        dist2[mask] = dist[mask] / diagonal2[mask]

        # Calculate dist1 where diagonal1 is not zero
        mask = diagonal1 != 0
        dist1[mask] = dist[mask] / diagonal1[mask]


        spatial = torch.cat((ious.unsqueeze(1),distZ.unsqueeze(1),dist1.unsqueeze(1),dist2.unsqueeze(1),areaZ.unsqueeze(1),area11.unsqueeze(1),area22.unsqueeze(1)),dim=1)
        cls1 = row[:,16]-1
        cls2 = row[:,17]-1
        cls1_int = cls1.cpu().numpy().astype(np.int32)
        cls2_int = cls2.cpu().numpy().astype(np.int32)

        cls1_feature =  one_hot_encode_batch(cls1_int , 48)   
        cls2_feature = one_hot_encode_batch(cls2_int , 48)  
        c = torch.cat((torch.tensor(cls1_feature).cuda(),torch.tensor(cls2_feature).cuda(),spatial),dim=1)

        return c,distI
    
        

    def sx_HBB(self,rel_pair_idxs,proposals):


        objlabel = proposals[0].extra_fields["labels"]
        objbox = proposals[0].bbox
        img_size = proposals[0].size
        print('img_size:', img_size)

        head_boxes = objbox[rel_pair_idxs[:, 0]]
        tail_boxes = objbox[rel_pair_idxs[:, 1]]
        head_labels = objlabel[rel_pair_idxs[:, 0]]
        tail_labels = objlabel[rel_pair_idxs[:, 1]]
        head_polys = self.convert_rect_to_points(head_boxes)
        tail_polys = self.convert_rect_to_points(tail_boxes)

        fu_pos_label = torch.cat((head_polys, tail_polys, head_labels.unsqueeze(1),tail_labels.unsqueeze(1)), dim=1)
        
        
        # feature, dist = self.calculate_HBB(fu_pos_label, img_size,head_boxes,tail_boxes)  

        
        batch_size = 1000000  # 设定 batch_size，具体数值可调整
        num_samples = head_boxes.shape[0]

        if num_samples  > 8000000:
            sub_losses = []
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                
                head_batch = head_boxes[start:end]
                tail_batch = tail_boxes[start:end]
                fu_batch = fu_pos_label[start:end]

                feature_batch, dist_batch = self.calculate_HBB(fu_batch, img_size, head_batch, tail_batch)  
                del dist_batch

                filter_fea = feature_batch.float()

                with torch.no_grad():
                        out1 = self.model1(filter_fea)
                        outputs2 = self.model2(out1)
                        loss1 = self.criterion(out1, filter_fea)
                        mean_loss1 = loss1.mean(dim=1)
                        loss2 = self.criterion(outputs2, filter_fea)
                        mean_loss2 = loss2.mean(dim=1)
                        sub_loss = (1/2)*mean_loss1+(1/2)*mean_loss2
                        sub_losses.append(sub_loss)
                        filter_fea = None  # Set the list element to None to release memory.

            # Concatenate the losses from the eight sub-tensors.
            loss = torch.cat(sub_losses, dim=0)



        else:
     
            feature, dist = self.calculate_HBB(fu_pos_label, img_size,head_boxes,tail_boxes)  
            filter_fea = feature.float()


            with torch.no_grad():
                out1 = self.model1(filter_fea)
                outputs2 = self.model2(out1)
                loss1 = self.criterion(out1, filter_fea)
                mean_loss1 = loss1.mean(dim=1)
                loss2 = self.criterion(outputs2, filter_fea)
                mean_loss2 = loss2.mean(dim=1)
                loss = (1/2)*mean_loss1+(1/2)*mean_loss2


        values, indices = torch.topk(loss, 10000, largest=False)  

        return [rel_pair_idxs[indices]]
        





                
                












    #     values, indices = torch.topk(loss, 10000, largest=False)  
        
      
        

    #     return [rel_pair_idxs[indices]]
    



