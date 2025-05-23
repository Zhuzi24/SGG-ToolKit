# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T


def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN  ## 600
        max_size = cfg.INPUT.MAX_SIZE_TRAIN  ## 1000
        flip_horizontal_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
        flip_vertical_prob = cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN # 0.0
        brightness = cfg.INPUT.BRIGHTNESS # 0.0
        contrast = cfg.INPUT.CONTRAST # 0.0
        saturation = cfg.INPUT.SATURATION # 0.0
        hue = cfg.INPUT.HUE # 0.0
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_horizontal_prob = 0.0
        flip_vertical_prob = 0.0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0

    to_bgr255 = cfg.INPUT.TO_BGR255 # true
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )
    color_jitter = T.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    transform = T.Compose(
        [
            color_jitter,
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(flip_horizontal_prob),
            T.RandomVerticalFlip(flip_vertical_prob),
            T.ToTensor(),
            normalize_transform,  ### build_transforms返回T.compose的对象，包含了各种图像处理操作
        ]
    )
    return transform
