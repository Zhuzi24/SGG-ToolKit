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
from argparse import ArgumentParser

from mmdet.apis import init_detector, show_result_pyplot

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
    parser.add_argument('--img_dir', default='data/DOTA/test/images/', help='Image file')
    parser.add_argument('--config', default='configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py', help='Config file')
    parser.add_argument('--checkpoint', default='work_dirs/oriented_rcnn/latest.pth', help='Checkpoint file')
    parser.add_argument('--out_dir', default='demo/output1/', help='Path to output file')
    parser.add_argument('--out',default='/media/dell/DATA/WLL/RSSGG/mmrotate/out/oriented_rcnn/outnew1.pkl', help='output result file in pickle format')
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
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args

outputs = []
def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    out_dir = args.out_dir

    images = get_img_file(args.img_dir)
    
    for img in images:
      print(img)
	 
      #img = mmcv.imread(image)    
      #result = inference_detector(model, img)

    
      result = inference_detector_by_patches(model, img, args.patch_sizes,
                                           args.patch_steps, args.img_ratios,
                                           args.merge_iou_thr)
      #import pdb; pdb.set_trace()
      outputs.append(result)   
      out_file = out_dir + img.split('/')[-1]
      show_result_pyplot(
        model,
        img,
        result,
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=out_file)
    if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
      

    # test a huge image by patches
    #import pdb; pdb.set_trace()                                       
    # show the results
    


if __name__ == '__main__':
    args = parse_args()
    main(args)
