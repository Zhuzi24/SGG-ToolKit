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
from mmcv.ops import box_iou_rotated
import cv2



class PPG(nn.Module):
    def __init__(self):
        super(PPG, self).__init__()
        self.bbox_overlaps = box_iou_rotated

        self.model1 = Autoencoder1(103, 25, 50, 50)
        self.model2 = Autoencoder2(103, 25, 50, 50)

        self.model1 = self.model1.cuda()
        self.model2 = self.model2.cuda()


        current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取脚本所在目录
        model_path = os.path.join(current_dir, 'best_model_OBB.pth')

        self.model1.load_state_dict(torch.load(model_path)['model_state_dict1'])
        self.model2.load_state_dict(torch.load(model_path)['model_state_dict2'])
        self.model1.eval()
        self.model2.eval()
        self.criterion = nn.MSELoss(reduction='none')
        

    def poly2obb_np_le90_8_batch_vectorized(self,polys_batch):
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
    
    def get_rotated_box_vertices_p(self, rboxes):
        if rboxes.dim() == 1:
            rboxes = rboxes.unsqueeze(0)
        N = rboxes.size(0)
        if N == 0:
            return rboxes.new_zeros((0, 8))
        x_ctr, y_ctr, width, height, angle = rboxes.t()
        tl_x, tl_y, br_x, br_y = -width * 0.5, -height * 0.5, width * 0.5, height * 0.5
        rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y]).reshape(2, 4, N).permute(2, 0, 1)
        sin, cos = torch.sin(angle), torch.cos(angle)
        M = torch.stack([cos, -sin, sin, cos]).reshape(2, 2, N).permute(2, 0, 1)
        polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
        polys[:, ::2] += x_ctr.unsqueeze(1)
        polys[:, 1::2] += y_ctr.unsqueeze(1)
        return polys

    def get_rotated_box_vertices(self, center_x, center_y, width, height, angle_degrees):
        """Convert oriented bounding boxes to polygons.
        Args:
            obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]    
        Returns:
            polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
        """
        rboxes = [center_x, center_y, width, height, angle_degrees]
        if len(rboxes) ==  5:
            rboxes = torch.Tensor([rboxes])
        N = rboxes.shape[0]
        if N == 0:
            return rboxes.new_zeros((rboxes.size(0), 8))
        x_ctr, y_ctr, width, height, angle = rboxes.select(1, 0), rboxes.select(1, 1), rboxes.select(1, 2), rboxes.select(1, 3), rboxes.select(1, 4)
        tl_x, tl_y, br_x, br_y = \
            -width * 0.5, -height * 0.5, \
            width * 0.5, height * 0.5
        if N != torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y],
                            dim=0).shape[-1]:
            N = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y],dim=0).shape[-1]
            print("check for here")
        rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y],dim=0).reshape(2, 4, N).permute(2, 0, 1)
        sin, cos = torch.sin(angle), torch.cos(angle)
        M = torch.stack([cos, -sin, sin, cos], dim=0).reshape(2, 2,N).permute(2, 0, 1)
        polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
        polys[:, ::2] += x_ctr.unsqueeze(1)
        polys[:, 1::2] += y_ctr.unsqueeze(1)
        polys = polys.contiguous()
        # polys = polys.numpy()
        return polys.tolist()[0]


    def calculate_rotated_box_area(self,boxes):
        # boxes: (num, 5) tensor, each row is (cx, cy, w, h, sita)
        _, _, w, h, _ = boxes.unbind(-1)
        area = w * h
        return area


    def calculate_spatialOriented(self, head5, tail5, ps1, img_size):

        def one_hot_encode_batch(num, alphabet_size):
            # cls: (batch_size,) array or list
            one_hot = np.eye(alphabet_size)[num]
            return one_hot
   
         
        spatial_cls = []
        w= img_size[0]
        h= img_size[1]
        row = ps1  # poly + poly + label + label 18
        # p1 = list(map(int,row[:8]))   
        # p2 = list(map(int,row[8:16]))
        poly1 = ps1[:,:8]
        poly2 = ps1[:,8:16]

        ious = torch.tensor(self.bbox_overlaps(  # 533,1003
        head5.float().cuda(),
        tail5.float(),aligned=True).cpu().numpy()).cuda()
        area1,area2 = self.calculate_rotated_box_area(head5), self.calculate_rotated_box_area(tail5)
        union_bbox = self.poly2obb_np_le90_8_batch_vectorized(ps1[:,:16])
        uarea = self.calculate_rotated_box_area(union_bbox)
        
        # not consider the case of zero area here
        area11=area1 / uarea.cuda()
        area22=area2 / uarea.cuda()
        areaZ = area1/area2
        # iou,area1,area2 = cal_iou(p1,p2)            
        # union_poly = np.concatenate((p1,p2))
        # union_poly = np.array(union_poly).reshape(8,2)
        # union_poly = Polygon(union_poly).convex_hull
        # uarea = union_poly.area    
            
        # if uarea == 0:
        #     print('box is None')
        #     area11=area1
        #     area22=area2          
        # else:
        #     area11=area1/uarea
        #     area22=area2/uarea
        # if area2 != 0:
        #     areaZ=area1/area2
        # else:           
        #     areaZ=area1

        cx1 = head5[:,0]
        cy1 = head5[:,1]
        cx2 = tail5[:,0]
        cy2 = tail5[:,1]
        
        #dist = math.sqrt((cx2-cx1)**2+(cy2-cy1)**2)
        dist = torch.sqrt((cx2-cx1)**2+(cy2-cy1)**2)
        distimg = math.sqrt(h**2+w**2)     
        distI = dist/distimg
        diagonal1 = torch.sqrt((head5[:,2])**2 + (head5[:,3])**2)
        diagonal2 = torch.sqrt((tail5[:,2])**2 + (tail5[:,3])**2)
        distZ = diagonal1/diagonal2
        dist2 = dist/diagonal2
        dist1 = dist/diagonal1

        spatial = torch.cat((ious.unsqueeze(1),distZ.unsqueeze(1),dist1.unsqueeze(1),dist2.unsqueeze(1),areaZ.unsqueeze(1),area11.unsqueeze(1),area22.unsqueeze(1)),dim=1)
        
        
        cls1 = row[:,16]-1
        cls2 = row[:,17]-1
        cls1_int = cls1.cpu().numpy().astype(np.int32)
        cls2_int = cls2.cpu().numpy().astype(np.int32)
        cls1_feature =  one_hot_encode_batch(cls1_int , 48)   
        cls2_feature = one_hot_encode_batch(cls2_int , 48)  

        c = torch.cat((torch.tensor(cls1_feature).cuda(),torch.tensor(cls2_feature).cuda(),spatial),dim=1)
        # feature= cls1_feature.tolist() + cls2_feature.tolist()
        # c = feature[:]    
        # c.extend(spatial) 
        return c,distI
    

    def sx_Oriented(self,rel_pair_idxs,proposals):


        objlabel = proposals[0].extra_fields["labels"]
        objbox = proposals[0].bbox
        img_size = proposals[0].size
        print('img_size:', img_size)

        head_boxes = objbox[rel_pair_idxs[:, 0]]
        tail_boxes = objbox[rel_pair_idxs[:, 1]]
        head_labels = objlabel[rel_pair_idxs[:, 0]]
        tail_labels = objlabel[rel_pair_idxs[:, 1]]
        head_polys = self.get_rotated_box_vertices_p(head_boxes)
        tail_polys = self.get_rotated_box_vertices_p(tail_boxes)

        fu_pos_label = torch.cat((head_polys, tail_polys, head_labels.unsqueeze(1),tail_labels.unsqueeze(1)), dim=1)

        # feature, dist = self.calculate_spatialOriented(head_boxes, tail_boxes, fu_pos_label, img_size)  

        batch_size = 1000000  # 设定 batch_size，具体数值可调整
        num_samples = head_boxes.shape[0]

        if num_samples  > 8000000:
            sub_losses = []
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                
                head_batch = head_boxes[start:end]
                tail_batch = tail_boxes[start:end]
                fu_batch = fu_pos_label[start:end]

                feature_batch, dist_batch = self.calculate_spatialOriented(head_batch, tail_batch, fu_batch, img_size)
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

            feature, dist = self.calculate_spatialOriented(head_boxes, tail_boxes, fu_pos_label, img_size)  
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
        

