# Copyright (c) OpenMMLab. All rights reserved.
"""Inference on huge images.

Example:
```
wget -P checkpoint https://download.openmmlab.com/mmrotate/v0.1.0/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth  # noqa: E501, E261.
python demo/huge_image_demo.py \
    demo/dota_demo.jpg \
    configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_v3.py \
    checkpoint/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth \
```
"""  # nowq
import mmcv
import os
import numpy as np
import math
from argparse import ArgumentParser
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
#from mmdet.apis import init_detector, show_result_pyplot

from mmrotate.apis import inference_detector_by_patches

def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                imagelist.append(os.path.join(parent, filename))
        
        return imagelist

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img_dir', default='data/DOTA/val/images/', help='Image file')
    parser.add_argument('--config', default='configs/oriented_rcnn/oriented_rcnn_swin_large_fpn_1x_dota_le90_IMP22k.py', help='Config file')
    parser.add_argument('--checkpoint', default='/media/dell/DATA/WLL/RSSGG/mmrotate/RS_LEAP_work_dirs/ms_oriented_rcnn_swinlarge_zengqiang/epoch_18.pth', help='Checkpoint file')
    parser.add_argument('--out_dir', default='/media/dell/DATA/WLL/RSSGG/mmrotate/results/imgsgdet/', help='Path to output file')
    #parser.add_argument('--out',default='/media/dell/DATA/WLL/RSSGG/mmrotate/out/oriented_rcnn/outnew1.pkl', help='output result file in pickle format')
    parser.add_argument(
        '--patch_sizes',
        type=int,
        nargs='+',
        default=[1024],
        help='The sizes of patches')
    parser.add_argument(
        '--patch_steps',
        type=int,
        nargs='+',
        default=[824],
        help='The steps between two patches')
    parser.add_argument(
        '--img_ratios',
        type=float,
        nargs='+',
        default=[1.0],
        help='Image resizing ratios for multi-scale detecting')
    parser.add_argument(
        '--merge_iou_thr',
        type=float,
        default=0.1,
        help='IoU threshould for merging results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'hrsc_classwise', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.1, help='bbox score threshold')
    args = parser.parse_args()
    return args

outputs = []
def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    out_dir = args.out_dir
    out_root_path='/media/dell/DATA/WLL/RSSGG/mmrotate/results/txtsgdet/'
    
    images = get_img_file(args.img_dir)
    for img in images:
        print(img)
	      #img = mmcv.imread(image)            
        result = inference_detector(model, img)
        #import pdb; pdb.set_trace()
        #result = inference_detector_by_patches(model, img, args.patch_sizes,
                                           #args.patch_steps, args.img_ratios,
                                           #args.merge_iou_thr)
        #print(result) 
        out_file = out_dir + img.split('/')[-1]
        #print(out_file)
        
        show_result_pyplot(
          model,
          img,
          result,
          palette=args.palette,
          score_thr=args.score_thr,
          out_file=out_file)
                
        bbox_result = result
        bboxes = np.vstack(bbox_result)  
        labels = [
            np.full(np.array(bbox).shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
      
        assert bboxes.ndim == 2
        assert labels.ndim == 1
        assert bboxes.shape[0] == labels.shape[0]
        assert bboxes.shape[1] == 6
      #scores = bboxes[:, -1]
##    scores_list = {0:0.5, 1:0.3, 2:0.1, 3:0.5, 4:0.3, 5:0.1}
      #scores_list = {0:0.3, 1:0.3, 2:0.3, 3:0.3, 4:0.3, 5:0.3, 6:0.3, 7:0.3, 8:0.3, 9:0.3, 10:0.3, 11:0.3, 12:0.3, 13:0.3, 14:0.3, 15:0.3, 16:0.3, 17:0.3, 18:0.3}
##    scores_list = {0:0.5, 1:0.1, 2:0.1, 3:0.3, 4:0.1, 5:0.1}
      #score_thrs = [scores_list[i] for i in labels]
      #score_thrs = np.array(score_thrs)
      #bboxes = bboxes[scores > score_thrs, :]
      #labels = labels[scores > score_thrs]
    
        results = []
        for bbox, label in zip(bboxes, labels):
            xc, yc, w, h, ag, p = bbox.tolist()
            wx, wy = w / 2 * math.cos(ag), w / 2 * math.sin(ag)
            hx, hy = -h / 2 * math.sin(ag), h / 2 * math.cos(ag)
            p1 = [xc - wx - hx, yc - wy - hy]
          #print(p1)
            p2 = [xc + wx - hx, yc + wy - hy]
          #print(p2)
            p3 = [xc + wx + hx, yc + wy + hy]
            p4 = [xc - wx + hx, yc - wy + hy]
            ps = list(p1 + p2 + p3 + p4)
          #print(ps)
          #ps = list(map(int,(p1 + p2 + p3 + p4)))
            ps += [label]
            ps += [p]
          
            results.append(ps)
      #print(11111111111111111)
      #print(results)      
        im=img.split('/')[-1]  
        count=0        
        with open(out_root_path+im.split('.')[0]+".txt", "w") as f:
            for line in results:
                if line[-1]<0.3: 
                    continue

                f.write(str(line) +'\n')
              #print('line: ', line)
                count += 1

            if count > 800:  
                break

            f.close()            
        
    #if args.out:
            #print(f'\nwriting results to {args.out}')
            #mmcv.dump(outputs, args.out)
      

    # test a huge image by patches
    #import pdb; pdb.set_trace()                                       
    # show the results
    


if __name__ == '__main__':
    args = parse_args()
    main(args)
