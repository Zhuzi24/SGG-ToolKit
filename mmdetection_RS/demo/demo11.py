from mmdet.apis import init_detector, inference_detector
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)

config_file = '/media/dell/DATA/WLL/RSSGG/mmdetection-master/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = '/media/dell/DATA/WLL/RSSGG/mmdetection-master/demo/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'


model = init_detector(config_file, checkpoint_file)


img = '/media/dell/DATA/WLL/RSSGG/mmdetection-master/demo/demo.jpg'
result = inference_detector(model, img)

show_result_pyplot(
        model,
        img,
        result,
        palette='coco',
        score_thr=0,
        out_file='/media/dell/DATA/WLL/RSSGG/mmdetection-master/demo/result.jpg')
#model.show_result(img, result, model.CLASSES)
'''

imgs = ['test1.jpg', 'test2.jpg']
for i, result in enumerate(inference_detector(model, imgs, device='cuda:0')):
    show_result(imgs[i], result, model.CLASSES, out_file='result_{}.jpg'.format(i))
'''