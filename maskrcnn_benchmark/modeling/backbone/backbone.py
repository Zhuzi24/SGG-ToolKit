# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict

from torch import nn
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from . import fpn as fpn_module
from . import resnet
from . import vgg




# #####
# from .alexnet import AlexNet
# # yapf: disable
# from .bricks import (ACTIVATION_LAYERS, CONV_LAYERS, NORM_LAYERS,
#                      PADDING_LAYERS, PLUGIN_LAYERS, UPSAMPLE_LAYERS,
#                      ContextBlock, Conv2d, Conv3d, ConvAWS2d, ConvModule,
#                      ConvTranspose2d, ConvTranspose3d, ConvWS2d,
#                      DepthwiseSeparableConvModule, GeneralizedAttention,
#                      HSigmoid, HSwish, Linear, MaxPool2d, MaxPool3d,
#                      NonLocal1d, NonLocal2d, NonLocal3d, Scale, Swish,
#                      build_activation_layer, build_conv_layer,
#                      build_norm_layer, build_padding_layer, build_plugin_layer,
#                      build_upsample_layer, conv_ws_2d, is_norm)
# from .builder import MODELS, build_model_from_cfg
# # yapf: enable
# from .resnet import ResNet, make_res_layer
# from .utils import (INITIALIZERS, Caffe2XavierInit, ConstantInit, KaimingInit,
#                     NormalInit, PretrainedInit, TruncNormalInit, UniformInit,
#                     XavierInit, bias_init_with_prob, caffe2_xavier_init,
#                     constant_init, fuse_conv_bn, get_model_complexity_info,
#                     initialize, kaiming_init, normal_init, trunc_normal_init,
#                     uniform_init, xavier_init)
# from .vgg import VGG, make_vgg_layer

__all__ = [
    'AlexNet', 'VGG', 'make_vgg_layer', 'ResNet', 'make_res_layer',
    'constant_init', 'xavier_init', 'normal_init', 'trunc_normal_init',
    'uniform_init', 'kaiming_init', 'caffe2_xavier_init',
    'bias_init_with_prob', 'ConvModule', 'build_activation_layer',
    'build_conv_layer', 'build_norm_layer', 'build_padding_layer',
    'build_upsample_layer', 'build_plugin_layer', 'is_norm', 'NonLocal1d',
    'NonLocal2d', 'NonLocal3d', 'ContextBlock', 'HSigmoid', 'Swish', 'HSwish',
    'GeneralizedAttention', 'ACTIVATION_LAYERS', 'CONV_LAYERS', 'NORM_LAYERS',
    'PADDING_LAYERS', 'UPSAMPLE_LAYERS', 'PLUGIN_LAYERS', 'Scale',
    'get_model_complexity_info', 'conv_ws_2d', 'ConvAWS2d', 'ConvWS2d',
    'fuse_conv_bn', 'DepthwiseSeparableConvModule', 'Linear', 'Conv2d',
    'ConvTranspose2d', 'MaxPool2d', 'ConvTranspose3d', 'MaxPool3d', 'Conv3d',
    'initialize', 'INITIALIZERS', 'ConstantInit', 'XavierInit', 'NormalInit',
    'TruncNormalInit', 'UniformInit', 'KaimingInit', 'PretrainedInit',
    'Caffe2XavierInit', 'MODELS', 'build_model_from_cfg'
]

####


@registry.BACKBONES.register("VGG-16")
def build_vgg_fpn_backbone(cfg):
    body = vgg.VGG16(cfg)
    out_channels = cfg.MODEL.VGG.VGG16_OUT_CHANNELS
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("R-50-C4")
@registry.BACKBONES.register("R-50-C5")
@registry.BACKBONES.register("R-101-C4")
@registry.BACKBONES.register("R-101-C5")
def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    return model

@registry.BACKBONES.register("R-50-FPN")
@registry.BACKBONES.register("R-101-FPN")
@registry.BACKBONES.register("R-152-FPN")
def build_resnet_fpn_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("R-50-FPN-RETINANET")
@registry.BACKBONES.register("R-101-FPN-RETINANET")
def build_resnet_fpn_p3p7_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
