# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.structures.image_list import to_image_list


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        if len(transposed_batch) == 3:
            images = to_image_list(transposed_batch[0], self.size_divisible)
            targets = transposed_batch[1]
            img_ids = transposed_batch[2]
            return images, targets, img_ids, None, None
        else:
            images = to_image_list(transposed_batch[0], self.size_divisible)
            targets = transposed_batch[1]
            img_ids = transposed_batch[2]
            
            ##
            imgs = to_image_list(transposed_batch[3], self.size_divisible)
            tar1 = transposed_batch[4]
            ##
            return images, targets, img_ids,  imgs , tar1


class BBoxAugCollator(object):
    """
    From a list of samples from the dataset,
    returns the images and targets.
    Images should be converted to batched images in `im_detect_bbox_aug`
    """

    def __call__(self, batch):
        return list(zip(*batch))

