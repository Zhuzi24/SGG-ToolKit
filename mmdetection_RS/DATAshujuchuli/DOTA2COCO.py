import dota_utils as util
import os
import cv2
import json

from PIL import ImageFile
from PIL import Image

# wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
#                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']

# wordname_16 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
#                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']
wordname_15 = ['ship','boat','crane','goods_yard','tank','storehouse','breakwater','dock','airplane','boarding_bridge','runway','taxiway','terminal','apron','gas_station','truck','car','truck_parking','car_parking','bridge','cooling_tower','chimney','vapor','smoke','genset','coal_yard']

def DOTA2COCOTrain(srcpath, destfile, cls_names, difficult='2'):
    # set difficult to filter '2', '1', or do not filter, set '-1'
    #import pdb; pdb.set_trace()

    imageparent = os.path.join(srcpath, 'images')
    labelparent = os.path.join(srcpath, 'annfiles')
    # labelparent = os.path.join(srcpath, 'annfiles')

    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = util.GetFileFromThisRootDir(labelparent)
        for file in filenames:
            basename = util.custombasename(file)
            # image_id = int(basename[1:])

            imagepath = os.path.join(imageparent, basename + '.png')
            img = cv2.imread(imagepath)
            height, width, c = img.shape

            single_image = {}
            single_image['file_name'] = basename + '.png'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            # annotations
            objects = util.parse_dota_poly2(file)
            #print(333333333333)
            #print(objects)
            for obj in objects:
                if obj['difficult'] == difficult:
                    print('difficult: ', difficult)
                    continue
                single_obj = {}
                single_obj['area'] = obj['area']
                single_obj['category_id'] = cls_names.index(obj['name']) + 1
                single_obj['segmentation'] = []
                single_obj['segmentation'].append(obj['poly'])
                single_obj['iscrowd'] = 0
                xmin, ymin, xmax, ymax = min(obj['poly'][0::2]), min(obj['poly'][1::2]), \
                                         max(obj['poly'][0::2]), max(obj['poly'][1::2])

                width, height = xmax - xmin, ymax - ymin
                single_obj['bbox'] = xmin, ymin, width, height
                single_obj['image_id'] = image_id
                data_dict['annotations'].append(single_obj)
                #print(11111)
                #print(data_dict)
                single_obj['id'] = inst_count
                inst_count = inst_count + 1
            image_id = image_id + 1
        json.dump(data_dict, f_out)

def DOTA2COCOVal111(srcpath, destfile, cls_names):
    imageparent = os.path.join(srcpath, 'images')
    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = util.GetFileFromThisRootDir(imageparent)
        for file in filenames:
            basename = util.custombasename(file)
            imagepath = os.path.join(imageparent, basename + '.png')
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            Image.MAX_IMAGE_PIXELS = None 
            img = Image.open(imagepath)
            height = img.height
            width = img.width

            single_image = {}
            single_image['file_name'] = basename + '.png'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            image_id = image_id + 1
        json.dump(data_dict, f_out)
        
def DOTA2COCOVal(srcpath, destfile, cls_names, difficult='2'):
    # set difficult to filter '2', '1', or do not filter, set '-1'

    imageparent = os.path.join(srcpath, 'images')
    labelparent = os.path.join(srcpath, 'annfiles')
    # labelparent = os.path.join(srcpath, 'annfiles')
    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = util.GetFileFromThisRootDir(labelparent)
        for file in filenames:
            basename = util.custombasename(file)
            # image_id = int(basename[1:])

            imagepath = os.path.join(imageparent, basename + '.png')
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            Image.MAX_IMAGE_PIXELS = None 
            img = Image.open(imagepath)
            height = img.height
            width = img.width
            single_image = {}
            single_image['file_name'] = basename + '.png'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            # annotations
            objects = util.parse_dota_poly2(file)
            for obj in objects:
                if obj['difficult'] == difficult:
                    #print('difficult: ', difficult)
                    continue
                single_obj = {}
                single_obj['area'] = obj['area']
                single_obj['category_id'] = cls_names.index(obj['name']) + 1
                single_obj['segmentation'] = []
                single_obj['segmentation'].append(obj['poly'])
                single_obj['iscrowd'] = 0
                xmin, ymin, xmax, ymax = min(obj['poly'][0::2]), min(obj['poly'][1::2]), \
                                         max(obj['poly'][0::2]), max(obj['poly'][1::2])

                width, height = xmax - xmin, ymax - ymin
                single_obj['bbox'] = xmin, ymin, width, height
                single_obj['image_id'] = image_id
                data_dict['annotations'].append(single_obj)
                #print(222222222222222)
                #print(data_dict)
                single_obj['id'] = inst_count
                inst_count = inst_count + 1
            image_id = image_id + 1
        json.dump(data_dict, f_out)

def DOTA2COCOTest(srcpath, destfile, cls_names, difficult='2'):
    # set difficult to filter '2', '1', or do not filter, set '-1'

    imageparent = os.path.join(srcpath, 'images')
    labelparent = os.path.join(srcpath, 'annfiles')
    # labelparent = os.path.join(srcpath, 'annfiles')
    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    for idex, name in enumerate(cls_names):
        single_cat = {'id': idex + 1, 'name': name, 'supercategory': name}
        data_dict['categories'].append(single_cat)

    inst_count = 1
    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = util.GetFileFromThisRootDir(labelparent)
        for file in filenames:
            basename = util.custombasename(file)
            # image_id = int(basename[1:])

            imagepath = os.path.join(imageparent, basename + '.png')
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            Image.MAX_IMAGE_PIXELS = None 
            img = Image.open(imagepath)
            height = img.height
            width = img.width
            single_image = {}
            single_image['file_name'] = basename + '.png'
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            # annotations
            objects = util.parse_dota_poly2(file)
            for obj in objects:
                if obj['difficult'] == difficult:
                    #print('difficult: ', difficult)
                    continue
                single_obj = {}
                single_obj['area'] = obj['area']
                single_obj['category_id'] = cls_names.index(obj['name']) + 1
                single_obj['segmentation'] = []
                single_obj['segmentation'].append(obj['poly'])
                single_obj['iscrowd'] = 0
                xmin, ymin, xmax, ymax = min(obj['poly'][0::2]), min(obj['poly'][1::2]), \
                                         max(obj['poly'][0::2]), max(obj['poly'][1::2])

                width, height = xmax - xmin, ymax - ymin
                single_obj['bbox'] = xmin, ymin, width, height
                single_obj['image_id'] = image_id
                data_dict['annotations'].append(single_obj)
                #print(222222222222222)
                #print(data_dict)
                single_obj['id'] = inst_count
                inst_count = inst_count + 1
            image_id = image_id + 1
        json.dump(data_dict, f_out)

if __name__ == '__main__':
    # DOTA2COCOTrain(r'/home/mmdetection_DOTA/data/dota1_1024_v2/trainval1024',
    #                r'/home/mmdetection_DOTA/data/dota1_1024_v2/trainval1024/DOTA_trainval1024.json',
    #                wordname_15)
    # DOTA2COCOTrain(r'/home/mmdetection_DOTA/data/dota1_1024_v2/trainval1024_ms',
    #                r'/home/mmdetection_DOTA/data/dota1_1024_v2/trainval1024_ms/DOTA_trainval1024_ms.json',
    #                wordname_15)
    # DOTA2COCOTest(r'/home/mmdetection_DOTA/data/dota1_1024_v2/test1024',
    #               r'/home/mmdetection_DOTA/data/dota1_1024_v2/test1024/DOTA_test1024.json',
    #               wordname_15)
    # DOTA2COCOTest(r'/home/mmdetection_DOTA/data/dota1_1024_v2/test1024_ms',
    #               r'/home/mmdetection_DOTA/data/dota1_1024_v2/test1024_ms/DOTA_test1024_ms.json',
    #               wordname_15)
    '''
    DOTA2COCOTrain(r'/media/dell/DATA/WLL/RSSGG/mmrotate/data/split_ms_dota/train',
                    r'/media/dell/DATA/WLL/RSSGG/mmdetection-master/data/RSLEAP650/annotations/DOTA_HB_train.json',
                    wordname_15)
    DOTA2COCOVal(r'/media/dell/DATA/WLL/RSSGG/mmrotate/data/split_ms_dota/val',
                    r'/media/dell/DATA/WLL/RSSGG/mmdetection-master/data/RSLEAP650/annotations/DOTA_HB_val.json',
                    wordname_15)

    DOTA2COCOTest(r'/media/dell/DATA/WLL/RSSGG/mmrotate/data/split_ms_dota/test',
                    r'/media/dell/DATA/WLL/RSSGG/mmdetection-master/data/RSLEAP650/annotations/DOTA_HB_test.json',
                    wordname_15)
    '''
    DOTA2COCOVal111(r'/media/dell/DATA/WLL/RSSGG/mmrotate/data/split_ms_dota/val',
                    r'/media/dell/DATA/WLL/RSSGG/mmdetection-master/data/RSLEAP650/annotations/DOTA_HB_test_wubox.json',
                    wordname_15)