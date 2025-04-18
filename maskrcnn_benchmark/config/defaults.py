# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os

from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the maximum image side during training will be
# INPUT.MAX_SIZE_TRAIN, while for testing it will be
# INPUT.MAX_SIZE_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.FLIP_AUG = False
_C.MODEL.RPN_ONLY = False
_C.MODEL.MASK_ON = False
_C.MODEL.RETINANET_ON = False
_C.MODEL.KEYPOINT_ON = False
_C.MODEL.ATTRIBUTE_ON = False
_C.MODEL.RELATION_ON = False
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
_C.MODEL.CLS_AGNOSTIC_BBOX_REG = False


_C.MIXUP = CN()
_C.MIXUP.FEAT_PATH = ''
_C.MIXUP.MIXUP_BG = False
_C.MIXUP.MIXUP_FG = False
_C.MIXUP.MIXUP_ADD_TAIL = False
_C.MIXUP.PREDICATE_LOSS_TYPE = None
_C.MIXUP.PREDICATE_USE_CURRI = False
_C.MIXUP.CONFIDENCE = False
_C.MIXUP.BG_LAMBDA = 0.0
_C.MIXUP.FG_LAMBDA = 0.0

# [4, 12, 21, 22, 23, 27, 33]
# [1, 3, 5, 6, 8, 9, 11, 18, 19, 29, 37, 38, 51]
# [2, 7, 10, 13, 14, 15, 16, 17, 20, 24, 25, 26, 28, 30, 31, 32, 34, 35, 36, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58]

# _C.HEAD_IDS = [4, 12, 21, 22, 23, 27, 33]
# _C.BODY_IDS = [1, 3, 5, 6, 8, 9, 11, 18, 19, 29, 37, 38, 51]
# _C.TAIL_IDS = [2, 7, 10, 13, 14, 15, 16, 17, 20, 24, 25, 26, 28, 30, 31, 32, 34, 35, 36, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58]

## 0101
_C.HEAD_IDS = [4, 12, 21, 22, 23, 27, 34]
_C.BODY_IDS = [1, 3, 5, 6, 8, 9, 10, 11, 16, 18, 19, 24, 29, 30, 31, 35, 37, 38, 39, 50, 51]
_C.TAIL_IDS = [2, 7, 13, 14, 15, 17, 20, 25, 26, 28, 32, 33, 36, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 52, 53, 54, 55, 56, 57, 58]

# If the WEIGHT starts with a catalog://, like :R-50, the code will look for
# the path in paths_catalog. Else, it will use it as the specified absolute
# path
_C.MODEL.WEIGHT = ""

# checkpoint of detector, for relation prediction
_C.MODEL.PRETRAINED_DETECTOR_CKPT = ""

#######
_C.WGAN = CN()
_C.WGAN.OUT = ""
_C.Flag = False
_C.Frebefor = False
_C.filter_method = "random_filter"
_C.RS_Leap =  True
_C.Type = "Large_RS_HBB"
_C.CFA_pre = None
_C.CONTRA = False
_C.Sema_F = False 
_C.Only_val = False
# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.MIN_SIZE_TRAIN = (800,)  # (800,)
# Maximum size of the side of the image during training
_C.INPUT.MAX_SIZE_TRAIN = 1333
# Size of the smallest side of the image during testing
_C.INPUT.MIN_SIZE_TEST = 800
# Maximum size of the side of the image during testing
_C.INPUT.MAX_SIZE_TEST = 1333
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [1., 1., 1.]
# Convert image to BGR format (for Caffe2 models), in range 0-255
_C.INPUT.TO_BGR255 = True

# Image ColorJitter
_C.INPUT.BRIGHTNESS = 0.0
_C.INPUT.CONTRAST = 0.0
_C.INPUT.SATURATION = 0.0
_C.INPUT.HUE = 0.0

_C.INPUT.VERTICAL_FLIP_PROB_TRAIN = 0.0

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for val, as present in paths_catalog.py
# Note that except dataset names, all remaining val configs reuse those of test
_C.DATASETS.VAL = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
_C.DATALOADER.SIZE_DIVISIBILITY = 0
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_C.DATALOADER.ASPECT_RATIO_GROUPING = True


# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

# The backbone conv body to use
# The string must match a function that is imported in modeling.model_builder
# (e.g., 'FPN.add_fpn_ResNet101_conv5_body' to specify a ResNet-101-FPN
# backbone)
_C.MODEL.BACKBONE.CONV_BODY = "R-50-C4"

# Add StopGrad at a specified stage so the bottom layers are frozen
_C.MODEL.BACKBONE.FREEZE_CONV_BODY_AT = 2


# ---------------------------------------------------------------------------- #
# FPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.FPN = CN()
_C.MODEL.FPN.USE_GN = False
_C.MODEL.FPN.USE_RELU = False


# ---------------------------------------------------------------------------- #
# Group Norm options
# ---------------------------------------------------------------------------- #
_C.MODEL.GROUP_NORM = CN()
# Number of dimensions per group in GroupNorm (-1 if using NUM_GROUPS)
_C.MODEL.GROUP_NORM.DIM_PER_GP = -1
# Number of groups in GroupNorm (-1 if using DIM_PER_GP)
_C.MODEL.GROUP_NORM.NUM_GROUPS = 32
# GroupNorm's small constant in the denominator
_C.MODEL.GROUP_NORM.EPSILON = 1e-5


# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.RPN = CN()
_C.MODEL.RPN.USE_FPN = False
_C.MODEL.RPN.RPN_MID_CHANNEL = 512
# Base RPN anchor sizes given in absolute pixels w.r.t. the scaled network input
_C.MODEL.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)
# Stride of the feature map that RPN is attached.
# For FPN, number of strides should match number of scales
_C.MODEL.RPN.ANCHOR_STRIDE = (16,)
# RPN anchor aspect ratios
_C.MODEL.RPN.ASPECT_RATIOS = (0.5, 1.0, 2.0)
# Remove RPN anchors that go outside the image by RPN_STRADDLE_THRESH pixels
# Set to -1 or a large value, e.g. 100000, to disable pruning anchors
_C.MODEL.RPN.STRADDLE_THRESH = 0
# Minimum overlap required between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
# ==> positive RPN example)
_C.MODEL.RPN.FG_IOU_THRESHOLD = 0.7
# Maximum overlap allowed between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
# ==> negative RPN example)
_C.MODEL.RPN.BG_IOU_THRESHOLD = 0.3
# Total number of RPN examples per image
_C.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
# Target fraction of foreground (positive) examples per RPN minibatch
_C.MODEL.RPN.POSITIVE_FRACTION = 0.5
# Number of top scoring RPN proposals to keep before applying NMS
# When FPN is used, this is *per FPN level* (not total)
_C.MODEL.RPN.PRE_NMS_TOP_N_TRAIN = 12000
_C.MODEL.RPN.PRE_NMS_TOP_N_TEST = 6000
# Number of top scoring RPN proposals to keep after applying NMS
_C.MODEL.RPN.POST_NMS_TOP_N_TRAIN = 2000
_C.MODEL.RPN.POST_NMS_TOP_N_TEST = 1000
# NMS threshold used on RPN proposals
_C.MODEL.RPN.NMS_THRESH = 0.7
# Proposal height and width both need to be greater than RPN_MIN_SIZE
# (a the scale used during training or inference)
_C.MODEL.RPN.MIN_SIZE = 0
# Number of top scoring RPN proposals to keep after combining proposals from
# all FPN levels
_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN = 2000
_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST = 2000
# Apply the post NMS per batch (default) or per image during training
# (default is True to be consistent with Detectron, see Issue #672)
_C.MODEL.RPN.FPN_POST_NMS_PER_BATCH = True
# Custom rpn head, empty to use default conv or separable conv
_C.MODEL.RPN.RPN_HEAD = "SingleConvRPNHead"


# ---------------------------------------------------------------------------- #
# ROI HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_HEADS = CN()
_C.MODEL.ROI_HEADS.USE_FPN = False
# Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
_C.MODEL.ROI_HEADS.FG_IOU_THRESHOLD = 0.5
# Overlap threshold for an RoI to be considered background
# (class = 0 if overlap in [0, BG_IOU_THRESHOLD))
_C.MODEL.ROI_HEADS.BG_IOU_THRESHOLD = 0.3

# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
_C.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)
# RoI minibatch size *per image* (number of regions of interest [ROIs])
# Total number of RoIs per training minibatch =
#   TRAIN.BATCH_SIZE_PER_IM * TRAIN.IMS_PER_BATCH
# E.g., a common configuration is: 512 * 2 * 8 = 8192
_C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
# Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
_C.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25

# Only used on test mode

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post processing steps (like NMS)
_C.MODEL.ROI_HEADS.SCORE_THRESH = 0.01
# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
_C.MODEL.ROI_HEADS.NMS = 0.3
_C.MODEL.ROI_HEADS.POST_NMS_PER_CLS_TOPN = 300
# Remove duplicated assigned labels for a single bbox in nms
_C.MODEL.ROI_HEADS.Frebefor = False 
# Maximum number of detections to return per image (100 is based on the limit
# established for the COCO dataset)
_C.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = 256


_C.MODEL.ROI_BOX_HEAD = CN()
_C.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
_C.MODEL.ROI_BOX_HEAD.PREDICTOR = "FastRCNNPredictor"
_C.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_BOX_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 81
# Hidden layer dimension when using an MLP for the RoI box head
_C.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM = 2048
# GN
_C.MODEL.ROI_BOX_HEAD.USE_GN = False
# Dilation
_C.MODEL.ROI_BOX_HEAD.DILATION = 1
_C.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM = 256
_C.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS = 4



_C.MODEL.ROI_ATTRIBUTE_HEAD = CN()
_C.MODEL.ROI_ATTRIBUTE_HEAD.FEATURE_EXTRACTOR = "FPN2MLPFeatureExtractor"
_C.MODEL.ROI_ATTRIBUTE_HEAD.PREDICTOR = "FPNPredictor"
_C.MODEL.ROI_ATTRIBUTE_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True
# Add attributes to each box
_C.MODEL.ROI_ATTRIBUTE_HEAD.USE_BINARY_LOSS = True
_C.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_LOSS_WEIGHT = 0.1
_C.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES = 201
_C.MODEL.ROI_ATTRIBUTE_HEAD.MAX_ATTRIBUTES = 10  # max number of attribute per bbox
_C.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_SAMPLE = True
_C.MODEL.ROI_ATTRIBUTE_HEAD.ATTRIBUTE_BGFG_RATIO = 3
_C.MODEL.ROI_ATTRIBUTE_HEAD.POS_WEIGHT = 5.0


_C.MODEL.ROI_MASK_HEAD = CN()
_C.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
_C.MODEL.ROI_MASK_HEAD.PREDICTOR = "MaskRCNNC4Predictor"
_C.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_MASK_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_MASK_HEAD.MLP_HEAD_DIM = 1024
_C.MODEL.ROI_MASK_HEAD.CONV_LAYERS = (256, 256, 256, 256)
_C.MODEL.ROI_MASK_HEAD.RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True
# Whether or not resize and translate masks to the input image.
_C.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS = False
_C.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD = 0.5
# Dilation
_C.MODEL.ROI_MASK_HEAD.DILATION = 1
# GN
_C.MODEL.ROI_MASK_HEAD.USE_GN = False

_C.MODEL.ROI_KEYPOINT_HEAD = CN()
_C.MODEL.ROI_KEYPOINT_HEAD.FEATURE_EXTRACTOR = "KeypointRCNNFeatureExtractor"
_C.MODEL.ROI_KEYPOINT_HEAD.PREDICTOR = "KeypointRCNNPredictor"
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_KEYPOINT_HEAD.MLP_HEAD_DIM = 1024
_C.MODEL.ROI_KEYPOINT_HEAD.CONV_LAYERS = tuple(512 for _ in range(8))
_C.MODEL.ROI_KEYPOINT_HEAD.RESOLUTION = 14
_C.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES = 17
_C.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True


_C.MODEL.ROI_RELATION_HEAD = CN()
### HETSGG
_C.MODEL.ROI_RELATION_HEAD.HETSGG = CN()
_C.MODEL.ROI_RELATION_HEAD.HETSGG.N_BASES = 6
_C.MODEL.ROI_RELATION_HEAD.HETSGG.H_DIM = 128
_C.MODEL.ROI_RELATION_HEAD.HETSGG.CATEGORY_FILE = 'VG-SGG-Category_Info'
###
# share box feature extractor should be set False for neural-motifs
_C.MODEL.ROI_RELATION_HEAD.PREDICTOR = "MotifPredictor"
_C.MODEL.ROI_RELATION_HEAD.FEATURE_EXTRACTOR = "RelationFeatureExtractor"
_C.MODEL.ROI_RELATION_HEAD.POOLING_ALL_LEVELS = True
_C.MODEL.ROI_RELATION_HEAD.NUM_CLASSES = 51
_C.MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE = 64
_C.MODEL.ROI_RELATION_HEAD.POSITIVE_FRACTION = 0.25
_C.MODEL.ROI_RELATION_HEAD.USE_GT_BOX = True
_C.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL = False
_C.MODEL.ROI_RELATION_HEAD.EMBED_DIM = 200
_C.MODEL.ROI_RELATION_HEAD.CONTEXT_DROPOUT_RATE = 0.2
_C.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM = 512
_C.MODEL.ROI_RELATION_HEAD.BGNN_CONTEXT_POOLING_DIM = 2048
_C.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM = 4096
_C.MODEL.ROI_RELATION_HEAD.CONTEXT_OBJ_LAYER = 1  # assert >= 1
_C.MODEL.ROI_RELATION_HEAD.CONTEXT_REL_LAYER = 1  # assert >= 1

_C.MODEL.ROI_RELATION_HEAD.TRANSFORMER = CN()
# for TransformerPredictor only
_C.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE = 0.1   
_C.MODEL.ROI_RELATION_HEAD.TRANSFORMER.OBJ_LAYER = 4        
_C.MODEL.ROI_RELATION_HEAD.TRANSFORMER.REL_LAYER = 2        
_C.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD = 8         
_C.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM = 2048     
_C.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM = 64         
_C.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM = 64         


######################################################
_C.MODEL.ROI_RELATION_HEAD.DPL = CN()
_C.MODEL.ROI_RELATION_HEAD.DPL.N_DIM = 128
_C.MODEL.ROI_RELATION_HEAD.DPL.ALPHA = 10
_C.MODEL.ROI_RELATION_HEAD.DPL.AVG_NUM_SAMPLE = 20
_C.MODEL.ROI_RELATION_HEAD.DPL.RADIUS = 1.0
_C.MODEL.ROI_RELATION_HEAD.DPL.FREQ_BASED_DIFF_N = False

######################################################

# graph trans pred
_C.MODEL.ROI_RELATION_HEAD.DUAL_TRANS = CN()
_C.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.USE_GTRANS_CONTEXT = True
_C.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.USE_REL_GRAPH = True

_C.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.EVAL_USE_FC = False

_C.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.USE_GRAPH_ENCODE = True

_C.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.GRAPH_ENCODE_STRATEGY = 'trans'  # cat_gcn

_C.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.TRANSFORMER = CN()
_C.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.TRANSFORMER.REL_LAYER = 2
_C.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.TRANSFORMER.NUM_HEAD = 8
_C.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.TRANSFORMER.DROPOUT_RATE = 0.1
_C.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.TRANSFORMER.KEY_DIM = 64
_C.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.TRANSFORMER.VAL_DIM = 64

_C.MODEL.ROI_RELATION_HEAD.DUAL_TRANS.REMOVE_BIAS = False
# end graph trans

# focal loss for relationship
_C.MODEL.ROI_RELATION_HEAD.USE_FOCAL_LOSS = False
_C.MODEL.ROI_RELATION_HEAD.FOCAL_LOSS = CN()
_C.MODEL.ROI_RELATION_HEAD.FOCAL_LOSS.GAMMA = 2.0
_C.MODEL.ROI_RELATION_HEAD.FOCAL_LOSS.ALPHA = 0.25
_C.MODEL.ROI_RELATION_HEAD.FOCAL_LOSS.SIZE_AVERAGE = True
# loss weight path for re-weight
_C.MODEL.ROI_RELATION_HEAD.LOSS_WEIGHT_PATH = ''
_C.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION = True

_C.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS = True
_C.MODEL.ROI_RELATION_HEAD.PROTREE_FILTER = False

### BGNN
_C.MODEL.ROI_RELATION_HEAD.BGNN_MODULE = CN()

_C.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.GRAPH_HIDDEN_DIM = 512 # the hidden dimension of graph model

_C.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.SHARE_PARAMETERS_EACH_ITER=True

_C.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.GRAPH_ITERATION_NUM = 3

_C.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.EDGE_FEATURES_REPRESENTATION = "union"  # obj_pair, 
# the feature representation for the relationship feature, can be the union features, instance pair features and thire fused features

_C.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.RELATION_CONFIDENCE_AWARE = False

_C.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.APPLY_GT = False

_C.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.MP_ON_VALID_PAIRS = False # graph will only message passing on edges filtered by the rel pn structure

_C.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.MP_VALID_PAIRS_NUM = 200 # the mp will take the top 150 relatedness score for mp

_C.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.ITERATE_MP_PAIR_REFINE = 0

_C.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.RELNESS_MP_WEIGHTING = False

_C.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.RELNESS_MP_WEIGHTING_SCORE_RECALIBRATION_METHOD = "minmax" # "learnable_scaling"

_C.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.LEARNABLE_SCALING_WEIGHT = (2.5, 0.03)  # (alpha, beta)

_C.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.SHARE_RELATED_MODEL_ACROSS_REFINE_ITER = False

_C.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.SPLIT_GRAPH4OBJ_REL = False

_C.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.GATING_WITH_RELNESS_LOGITS = False

_C.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.SKIP_CONNECTION_ON_OUTPUT= False # add the skip connection on the graph output and initial input

# ### bias module
_C.MODEL.ROI_RELATION_HEAD.BIAS_MODULE = CN()
_C.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.EPSILON = 1e-3
_C.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.USE_PENALTY = False
_C.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.DROPOUT = 0.0

_C.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.PENALTY_BIAS = CN()
_C.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.PENALTY_BIAS.PENALTY_THRESHOLD = 0.5
_C.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.PENALTY_BIAS.PENALTY_WEIGHT = 0.1
_C.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.PENALTY_BIAS.SCALE_WEIGHT = 1.0
_C.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.PENALTY_BIAS.BG_DEFAULT_VALUE = 0.02
_C.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.PENALTY_BIAS.PENALTY_TYPE = 'static'  # 'weight' 'count' 'set_low' 'tail_low'
_C.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.PENALTY_BIAS.PENALTY_FUSION_WEIGHTS = [1.0, 1.0]
_C.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.PENALTY_BIAS.PENALTY_K = 10
_C.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.PENALTY_BIAS.PENALTY_EPSILON = 1e-3
_C.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.PENALTY_BIAS.WEIGHT_PATH = ''

_C.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.PENALTY_BIAS.CB_CLS_FUSION_TYPE = 'sum'  # 'mean', 'iter_l2'
_C.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.PENALTY_BIAS.CLS_TRANS = 'none'  # 'linear'
_C.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.PENALTY_BIAS.CB_CLS_FUSION_WEIGHT = 0.8  # for cb_cls

_C.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.PENALTY_BIAS.EVAL_WITH_PENALTY = False
_C.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.PENALTY_BIAS.USE_NEG_PENALTY = False

_C.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.PENALTY_BIAS.POSSIBLE_BIAS_DEFAULT_VALUE = 1.0
_C.MODEL.ROI_RELATION_HEAD.BIAS_MODULE.PENALTY_BIAS.POSSIBLE_BIAS_THRESHOLD = 100.0
# ### end bias module

_C.MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS = False
_C.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION = True
_C.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS = True
_C.MODEL.ROI_RELATION_HEAD.REQUIRE_BOX_OVERLAP = True
_C.MODEL.ROI_RELATION_HEAD.NUM_SAMPLE_PER_GT_REL = 4  # when sample fg relationship from gt, the max number of corresponding proposal pairs

# in sgdet, to make sure the detector won't missing any ground truth bbox, 
# we add grount truth box to the output of RPN proposals during Training
_C.MODEL.ROI_RELATION_HEAD.ADD_GTBOX_TO_PROPOSAL_IN_TRAIN = False


_C.MODEL.ROI_RELATION_HEAD.CAUSAL = CN()
# direct and indirect effect analysis
_C.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS = False
# Fusion
_C.MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE = 'sum'
# causal context feature layer
_C.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER = 'motifs'
# separate spatial in union feature
_C.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL = False

_C.MODEL.ROI_RELATION_HEAD.CAUSAL.SPATIAL_FOR_VISION = False

_C.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE = 'none' # 'TDE', 'TIE', 'TE'

# proportion of predicates
_C.MODEL.ROI_RELATION_HEAD.REL_PROP = [0.01858, 0.00057, 0.00051, 0.00109, 0.00150, 0.00489, 0.00432, 0.02913, 0.00245, 0.00121, 
                                       0.00404, 0.00110, 0.00132, 0.00172, 0.00005, 0.00242, 0.00050, 0.00048, 0.00208, 0.15608,
                                       0.02650, 0.06091, 0.00900, 0.00183, 0.00225, 0.00090, 0.00028, 0.00077, 0.04844, 0.08645,
                                       0.31621, 0.00088, 0.00301, 0.00042, 0.00186, 0.00100, 0.00027, 0.01012, 0.00010, 0.01286,
                                       0.00647, 0.00084, 0.01077, 0.00132, 0.00069, 0.00376, 0.00214, 0.11424, 0.01205, 0.02958]

_C.MODEL.VGG = CN()
_C.MODEL.VGG.VGG16_OUT_CHANNELS= 512
# ---------------------------------------------------------------------------- #
# ResNe[X]t options (ResNets = {ResNet, ResNeXt}
# Note that parts of a resnet may be used for both the backbone and the head
# These options apply to both
# ---------------------------------------------------------------------------- #
_C.MODEL.RESNETS = CN()

# Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
_C.MODEL.RESNETS.NUM_GROUPS = 1

# Baseline width of each group
_C.MODEL.RESNETS.WIDTH_PER_GROUP = 64

# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
_C.MODEL.RESNETS.STRIDE_IN_1X1 = True

# Residual transformation function
_C.MODEL.RESNETS.TRANS_FUNC = "BottleneckWithFixedBatchNorm"
# ResNet's stem function (conv1 and pool1)
_C.MODEL.RESNETS.STEM_FUNC = "StemWithFixedBatchNorm"

# Apply dilation in stage "res5"
_C.MODEL.RESNETS.RES5_DILATION = 1

_C.MODEL.RESNETS.BACKBONE_OUT_CHANNELS = 256 * 4
_C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
_C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64

_C.MODEL.RESNETS.STAGE_WITH_DCN = (False, False, False, False)
_C.MODEL.RESNETS.WITH_MODULATED_DCN = False
_C.MODEL.RESNETS.DEFORMABLE_GROUPS = 1


# ---------------------------------------------------------------------------- #
# RetinaNet Options (Follow the Detectron version)
# ---------------------------------------------------------------------------- #
_C.MODEL.RETINANET = CN()

# This is the number of foreground classes and background.
_C.MODEL.RETINANET.NUM_CLASSES = 81

# Anchor aspect ratios to use
_C.MODEL.RETINANET.ANCHOR_SIZES = (32, 64, 128, 256, 512)
_C.MODEL.RETINANET.ASPECT_RATIOS = (0.5, 1.0, 2.0)
_C.MODEL.RETINANET.ANCHOR_STRIDES = (8, 16, 32, 64, 128)
_C.MODEL.RETINANET.STRADDLE_THRESH = 0

# Anchor scales per octave
_C.MODEL.RETINANET.OCTAVE = 2.0
_C.MODEL.RETINANET.SCALES_PER_OCTAVE = 3

# Use C5 or P5 to generate P6
_C.MODEL.RETINANET.USE_C5 = True

# Convolutions to use in the cls and bbox tower
# NOTE: this doesn't include the last conv for logits
_C.MODEL.RETINANET.NUM_CONVS = 4

# Weight for bbox_regression loss
_C.MODEL.RETINANET.BBOX_REG_WEIGHT = 4.0

# Smooth L1 loss beta for bbox regression
_C.MODEL.RETINANET.BBOX_REG_BETA = 0.11

# During inference, #locs to select based on cls score before NMS is performed
# per FPN level
_C.MODEL.RETINANET.PRE_NMS_TOP_N = 1000

# IoU overlap ratio for labeling an anchor as positive
# Anchors with >= iou overlap are labeled positive
_C.MODEL.RETINANET.FG_IOU_THRESHOLD = 0.5

# IoU overlap ratio for labeling an anchor as negative
# Anchors with < iou overlap are labeled negative
_C.MODEL.RETINANET.BG_IOU_THRESHOLD = 0.4

# Focal loss parameter: alpha
_C.MODEL.RETINANET.LOSS_ALPHA = 0.25

# Focal loss parameter: gamma
_C.MODEL.RETINANET.LOSS_GAMMA = 2.0

# Prior prob for the positives at the beginning of training. This is used to set
# the bias init for the logits layer
_C.MODEL.RETINANET.PRIOR_PROB = 0.01

# Inference cls score threshold, anchors with score > INFERENCE_TH are
# considered for inference
_C.MODEL.RETINANET.INFERENCE_TH = 0.05

# NMS threshold used in RetinaNet
_C.MODEL.RETINANET.NMS_TH = 0.4


# ---------------------------------------------------------------------------- #
# FBNet options
# ---------------------------------------------------------------------------- #
_C.MODEL.FBNET = CN()
_C.MODEL.FBNET.ARCH = "default"
# custom arch
_C.MODEL.FBNET.ARCH_DEF = ""
_C.MODEL.FBNET.BN_TYPE = "bn"
_C.MODEL.FBNET.SCALE_FACTOR = 1.0
# the output channels will be divisible by WIDTH_DIVISOR
_C.MODEL.FBNET.WIDTH_DIVISOR = 1
_C.MODEL.FBNET.DW_CONV_SKIP_BN = True
_C.MODEL.FBNET.DW_CONV_SKIP_RELU = True

# > 0 scale, == 0 skip, < 0 same dimension
_C.MODEL.FBNET.DET_HEAD_LAST_SCALE = 1.0
_C.MODEL.FBNET.DET_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_C.MODEL.FBNET.DET_HEAD_STRIDE = 0

# > 0 scale, == 0 skip, < 0 same dimension
_C.MODEL.FBNET.KPTS_HEAD_LAST_SCALE = 0.0
_C.MODEL.FBNET.KPTS_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_C.MODEL.FBNET.KPTS_HEAD_STRIDE = 0

# > 0 scale, == 0 skip, < 0 same dimension
_C.MODEL.FBNET.MASK_HEAD_LAST_SCALE = 0.0
_C.MODEL.FBNET.MASK_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_C.MODEL.FBNET.MASK_HEAD_STRIDE = 0

# 0 to use all blocks defined in arch_def
_C.MODEL.FBNET.RPN_HEAD_BLOCKS = 0
_C.MODEL.FBNET.RPN_BN_TYPE = ""


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 40000

_C.SOLVER.BASE_LR = 0.002
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0
_C.SOLVER.CLIP_NORM = 5.0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.SCHEDULE = CN()
_C.SOLVER.SCHEDULE.TYPE = "WarmupMultiStepLR"  # "WarmupReduceLROnPlateau"
# the following paramters are only used for WarmupReduceLROnPlateau
_C.SOLVER.SCHEDULE.PATIENCE = 2
_C.SOLVER.SCHEDULE.THRESHOLD = 1e-4
_C.SOLVER.SCHEDULE.COOLDOWN = 1
_C.SOLVER.SCHEDULE.FACTOR = 0.5
_C.SOLVER.SCHEDULE.MAX_DECAY_STEP = 7


_C.SOLVER.CHECKPOINT_PERIOD = 2500

_C.SOLVER.GRAD_NORM_CLIP = 5.0

_C.SOLVER.PRINT_GRAD_FREQ = 5000
# whether validate and validate period
_C.SOLVER.TO_VAL = True
_C.SOLVER.PRE_VAL = True
_C.SOLVER.VAL_PERIOD = 2500

# update schedule
# when load from a previous model, if set to True
# only maintain the iteration number and all the other settings of the 
# schedule will be changed
_C.SOLVER.UPDATE_SCHEDULE_DURING_LOAD = False

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 16

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.EXPECTED_RESULTS = []
_C.TEST.EXPECTED_RESULTS_SIGMA_TOL = 4
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST.IMS_PER_BATCH = 8
# Number of detections per image
_C.TEST.DETECTIONS_PER_IMG = 100

# ---------------------------------------------------------------------------- #
# Test-time augmentations for bounding box detection
# See configs/test_time_aug/e2e_mask_rcnn_R-50-FPN_1x.yaml for an example
# ---------------------------------------------------------------------------- #
_C.TEST.BBOX_AUG = CN()

# Enable test-time augmentation for bounding box detection if True
_C.TEST.BBOX_AUG.ENABLED = False

# Horizontal flip at the original scale (id transform)
_C.TEST.BBOX_AUG.H_FLIP = False

# Each scale is the pixel size of an image's shortest side
_C.TEST.BBOX_AUG.SCALES = ()

# Max pixel size of the longer side
_C.TEST.BBOX_AUG.MAX_SIZE = 4000

# Horizontal flip at each scale
_C.TEST.BBOX_AUG.SCALE_H_FLIP = False

_C.TEST.SAVE_PROPOSALS = False
# Settings for relation testing
_C.TEST.RELATION = CN()
_C.TEST.RELATION.MULTIPLE_PREDS = False
_C.TEST.RELATION.IOU_THRESHOLD = 0.5
_C.TEST.RELATION.REQUIRE_OVERLAP = True
# when predict the label of bbox, run nms on each cls
_C.TEST.RELATION.LATER_NMS_PREDICTION_THRES = 0.3 
# synchronize_gather, used for sgdet, otherwise test on multi-gpu will cause out of memory
_C.TEST.RELATION.SYNC_GATHER = False

_C.TEST.ALLOW_LOAD_FROM_CACHE = True


_C.TEST.CUSTUM_EVAL = False
_C.TEST.CUSTUM_PATH = '.'

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."
_C.DETECTED_SGG_DIR = "."
_C.GLOVE_DIR = "."

_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
_C.PATHS_DATA = os.path.join(os.path.dirname(__file__), "../data/datasets")

# ---------------------------------------------------------------------------- #
# Precision options
# ---------------------------------------------------------------------------- #

# Precision of input, allowable: (float32, float16)
_C.DTYPE = "float32"

# Enable verbosity in apex.amp
_C.AMP_VERBOSE = False

_C.DEBUG = False
_C.TEST.DEBUG = False
# cc allow debug config delete
_C.set_new_allowed(True)

_C.USE_PREDCLS_FEATURE = True

_C.FG_HEAD = False
_C.FG_BODY = False
_C.FG_TAIL = False

_C.BG_HEAD = False
_C.BG_BODY = False
_C.BG_TAIL = False

_C.CL_HEAD = False
_C.CL_BODY = False
_C.CL_TAIL = False

_C.CONTRA = False
_C.PKO = False
_C.SELECT_DATASET = 'VG'
_C.EXTRACT_GROUP = 'body'


_C.MODEL.ROI_RELATION_HEAD.GPSNET_MODULE = CN()

_C.MODEL.ROI_RELATION_HEAD.GPSNET_MODULE.GRAPH_ITERATION_NUM = 2

_C.MODEL.ROI_RELATION_HEAD.GPSNET_MODULE.ITERATE_MP_PAIR_REFINE = 2

_C.MODEL.ROI_RELATION_HEAD.GPSNET_MODULE.MP_ON_VALID_PAIRS = False # graph will only message passing on edges filtered by the rel pn structure

_C.MODEL.ROI_RELATION_HEAD.GPSNET_MODULE.MP_VALID_PAIRS_NUM = 200 # the mp will take the top 150 relatedness score for mp

_C.MODEL.ROI_RELATION_HEAD.GPSNET_MODULE.RELNESS_MP_WEIGHTING = False

_C.MODEL.ROI_RELATION_HEAD.GPSNET_MODULE.GRAPH_HIDDEN_DIM = 512

_C.MODEL.ROI_RELATION_HEAD.CLASS_BALANCE_LOSS = False