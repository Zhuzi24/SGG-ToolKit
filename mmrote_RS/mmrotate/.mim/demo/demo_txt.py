# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

import mmrotate  # noqa: F401
import os

def parse_args():
    parser = ArgumentParser()
    #parser.add_argument('--img_path',default='/project/luojunwei/Datasets/Shanghai_add/Shanghai_Bridges/test/images', help='Image file')
    # parser.add_argument('--img_path',default='/project/luojunwei/test4/mmrotate/demo_shanghai/img512', help='Image file')
    # parser.add_argument('--img_path',default='/scratch/luojunwei/Dataset/WaterBody_split/images', help='Image file')
    #parser.add_argument('--img_path',default='/scratch/luojunwei/Dataset/WaterBody_MultiScale/WaterBody_split_6400/images', help='Image file')
    parser.add_argument('--img_path',default='/media/dell/DATA/WLL/RSSGG/mmrotate/data/DOTA/test/images', help='Image file')
   
    
    # parser.add_argument('--config',default='oriented_reppoints_r50_fpn_1x_dota_le135.py', help='Config file')
    #parser.add_argument('--checkpoint',default='epoch_14.pth',  help='Checkpoint file') 
    # parser.add_argument('--config',default='/project/luojunwei/test4/mmrotate/tools/AllBridge_train2/oriented_reppoints_r50_fpn_1x_dota_le135.py', help='Config file')
    # parser.add_argument('--checkpoint',default='/project/luojunwei/test4/mmrotate/tools/AllBridge_train2/oriented_reppoints_r50_fpn_1x_dota_le135.py/epoch_14.pth',  help='Checkpoint file') #
    # parser.add_argument('--checkpoint', default='/project/luojunwei/test4/mmrotate/tools/Bridge_Subset/oriented_reppoints_r50_fpn_1x_dota_le135/epoch_16.pth',help='checkpoint file')    
    # parser.add_argument('--checkpoint', default='/project/luojunwei/test4/mmrotate/tools/Bridge_Subset/oriented_reppoints_r50_fpn_1x_dota_le135/epoch_14.pth',help='checkpoint file')
    #parser.add_argument('--config', default='/project/luojunwei/test4/mmrotate/configs/oriented_reppoints/oriented_reppoints_r50_fpn_1x_dota_le135_ms.py',help='train config file path')
    parser.add_argument('--config', default='configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py',help='train config file path')
    
    parser.add_argument('--checkpoint', default='work_dirs/oriented_rcnn/latest.pth',help='checkpoint file')
    

    parser.add_argument('--out-file', default='result/result.jpg', help='Path to output file') #
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
        #'--device', default='cpu', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='dota',
        choices=['dota', 'sar', 'hrsc', 'hrsc_classwise', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main(args):

    parent_path=args.img_path
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    
    i=0
    for img in os.listdir(parent_path):
        i+=1
        # if i<664:
        #     continue
        print('processing-- '+img+'--')
        #out_file = 'result/' + img
        out_root_path='/media/dell/DATA/WLL/RSSGG/mmrotate/result/img/'
        out_file = out_root_path + img
    
        # test a single image
        imgname=img
        img=os.path.join(parent_path,img)
        result = inference_detector(model, img)


        # show the results
        # show_result_pyplot(
        #     model,
        #     img,
        #     result,
        #     palette=args.palette,
        #     score_thr=args.score_thr,
        #     out_file=out_file)

        count=0
        
        with open(out_root_path+imgname.split('.')[0]+".txt", "w") as f:
            for res in result:
                for line in res:
                    if line[-1]<0.3: 
                        continue

                    f.write(str(line)+'\n')
                    #print('line: ', line)
                    count += 1

                if count > 30:  
                    break

            f.close()
    
    


if __name__ == '__main__':
    args = parse_args()
    main(args)
