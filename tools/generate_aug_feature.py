# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import time
import datetime
import numpy as np
import copy
import torch
from torch.nn.utils import clip_grad_norm_

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.checkpoint import clip_grad_norm
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, all_gather
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger, debug_print
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from tqdm import tqdm

from IPython.core.display import display
from matplotlib.pyplot import imshow
from PIL import Image, ImageDraw
from tqdm import tqdm
from mmrotate.datasets import build_dataset
from mmrotate.models import build_detector

from mmdet.datasets import build_dataset as b_data
from mmdet.models import build_detector as b_det
from mmdet.datasets import build_dataloader as b_loader


from mmcv import Config, DictAction
# from mmdet.datasets import (build_dataloader, build_dataset,
#                             replace_ImageToTensor)

from mmdet.datasets import (build_dataset,
                            replace_ImageToTensor)
from maskrcnn_benchmark.modeling.detector.b_test import build_dataloader


from mmdet.apis import init_random_seed, set_random_seed
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner)
from collections import OrderedDict
from mmrotate.core.evaluation.eval_map import eval_rbbox_map
import mmcv
from mmcv.image import tensor2imgs
import matplotlib.pyplot as plt

from mmdet.core import encode_mask_results
from mmcv.cnn import fuse_conv_bn

# from maskrcnn_benchmark.modeling.roi_heads.relation_head.embe_1019 import weight
# # from maskrcnn_benchmark.modeling.roi_heads.relation_head.gen_wei import GEN_wei
from wgan2 import GAN
# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
# torch.cuda.set_device(1)
# torch.cuda.set_device(3)
from torch.optim import  lr_scheduler
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')
from numpy import random
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
import sys
seed_torch()
import torch.distributed as dist

from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

CV  = None

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

def parse_args_OBB(mmcf = None,mmwei = None):

    parser = argparse.ArgumentParser(description='Train a detector')

   # parser.add_argument('--config', default='/media/dell/data1/WTZ/SGG_Frame/mmrote_RS/configs/oriented_rcnn/oriented_rcnn_swin_large_fpn_1x_dota_le90_IMP22k.py', help='train config file path')
    parser.add_argument('--work-dir', default='/media/dell/data1/WTZ/SGG_Frame/mmrote_RS/out', help='the dir to save logs and models')
   
   # for large RS
    parser.add_argument('--config', default=mmcf, help='train config file path')
   
    #  for small 
    # parser.add_argument('--config', default='/media/dell/data1/WTZ/SGG_Frame/configs/RS_small/oriented_rcnn_swin_large_fpn_1x_dota_le90_IMP22k.py', help='train config file path')
    # parser.add_argument('--checkpoint',default='/media/dell/data1/WTZ/SGG_Frame/configs/RS_small/epoch_12.pth', help='checkpoint file')
    
    parser.add_argument('--checkpoint',default= mmwei, help='checkpoint file')
    # /media/dell/data1/WTZ/SGG_Frame/checkpoints/PENET_RS_Sgcls/20000.pth/media/dell/data1/WTZ/SGG_Frame/checkpoints/1219/latest.pth 
    #  "/media/dell/data1/WTZ/SGG_Frame/checkpoints/1227/18000.pth"   /media/dell/data1/WTZ/SGG_Frame/checkpoints/PENET_RS_SGDET_use_box_nms/1500.pth
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    ####  val
    # parser.add_argument('--config', default='/media/dell/data1/WTZ/SGG_Frame/mmrote_RS/checkpoints/1219/oriented_rcnn_swin_large_fpn_1x_dota_le90_IMP22k.py', help='test config file path')
    # parser.add_argument('--checkpoint',default='/media/dell/data1/WTZ/SGG_Frame/mmrote_RS/checkpoints/1219/latest.pth', help='checkpoint file')
    # parser.add_argument(
    #     '--work-dir',
    #     help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out',default='/media/dell/data1/WTZ/SGG_Frame/mmrote_RS/checkpoints/1219/oriented_rcnn/outshiyan.pkl', help='output result file in pickle format')
    #parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')

    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
                default='mAP',
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', default='/media/dell/data1/WTZ/SGG_Frame/mmrote_RS/checkpoints/1219', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')

    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    #####

    args = parser.parse_args(args=[])
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def parse_args_HBB(mmcf = None,mmwei = None):
    parser = argparse.ArgumentParser(description='Train a detector')

    # for large 
    parser.add_argument('--config',default = mmcf, help='train config file path')
    
    # for small  '/media/dell/data1/WTZ/SGG_Frame/configs/RSHBB/faster_rcnn_r50_fpn_1x_RS650.py'
  #  parser.add_argument('--config',default='/media/dell/data1/WTZ/SGG_Frame/configs/RS_HBB_small/HBB_small_faster_rcnn_r50_fpn_1x_RS650.py', help='train config file path')
    
    parser.add_argument('--checkpoint',default= mmwei , help='')
    # /media/dell/data1/WTZ/SGG_Frame/mmdetection_RS/RSLEAP_HBB/epoch_15.pth
    #parser.add_argument('--checkpoint',default='/media/dell/data1/WTZ/SGG_Frame/checkpoints/HBB/RTPB/SGdet/18000.pth', help='')
    # parser.add_argument('--checkpoint',default='/media/dell/data1/WTZ/SGG_Frame/checkpoints/HBB/PEnet/PENET_RS_Sgcls/20000.pth', help='')
    # parser.add_argument('--checkpoint',default='/media/dell/data1/WTZ/SGG_Frame/checkpoints/HBB/PENET_RS_NEW_15/20000.pth', help='')

    parser.add_argument('--work-dir', default='work-dir-RSLEAP',help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()

    parser.add_argument('--out',default="/media/dell/data1/WTZ/SGG_Frame/mmdetection_RS/RSLEAP_HBB/15.pkl", help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')

    parser.add_argument(
        '--eval',
        type=str,
        default="bbox",
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')

    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    args = parser.parse_args(args=[])
    
    # args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def generate_aug_results(cfg, local_rank, distributed, logger, mix_up_path, mmcf = None,mmwei = None):

    logger.info("***********************TYPE***********************")
    logger.info("do" + cfg.Type)

    head_rel_ids = cfg.HEAD_IDS
    body_rel_ids = cfg.BODY_IDS
    tail_rel_ids = cfg.TAIL_IDS
    filter_rel_ids_dicts = dict(zip(['head', 'body', 'tail'], [head_rel_ids, body_rel_ids, tail_rel_ids]))
 
    logger.info("***********************TYPE***********************")
    logger.info("do  " + cfg.Type)

    if cfg.Type != "CV": 
        logger.info("config: " + mmcf)
        logger.info("prewei: " + mmwei)

    if cfg.Type == "CV": 

        logger.info("***********************Step 1: model  construction***********************")
        print('\n')

        debug_print(logger, 'CV construction for faster rcnn -- bbox')
        model = build_detection_model(cfg)
        debug_print(logger, 'end model construction --- CV construction for faster rcnn -- bbox')

        logger.info('modules that should be always set in eval mode, their eval() method should be called after model.train() is called')
        eval_modules = (model.rpn, model.backbone, model.roi_heads.box,)
        fix_eval_modules(eval_modules)
        logger.info(" done ! param.requires_grad = False for model.rpn, model.backbone, model.roi_heads.box")

        # NOTE, we slow down the LR of the layers start with the names in slow_heads
        if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "IMPPredictor":
            slow_heads = ["roi_heads.relation.box_feature_extractor",
                        "roi_heads.relation.union_feature_extractor.feature_extractor", ]
        else:
            slow_heads = []

        # load pretrain layers to new layers
        load_mapping = {"roi_heads.relation.box_feature_extractor": "roi_heads.box.feature_extractor",
                        "roi_heads.relation.union_feature_extractor.feature_extractor": "roi_heads.box.feature_extractor"}

        if cfg.MODEL.ATTRIBUTE_ON:
            load_mapping["roi_heads.relation.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"
            load_mapping[
                "roi_heads.relation.union_feature_extractor.att_feature_extractor"] = "roi_heads.attribute.feature_extractor"
        
        logger.info("print model parameters")
        logger.info(show_params_status(model))
        logger.info("done !!! print model parameters")

        device = cfg.MODEL.DEVICE # Btorch.device("cuda:0") #
        model.to(device)

        num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
        num_batch = cfg.SOLVER.IMS_PER_BATCH
        optimizer = make_optimizer(cfg, model, logger, slow_heads=slow_heads, slow_ratio=10.0, rl_factor=float(num_batch))
        scheduler = make_lr_scheduler(cfg, optimizer, logger)
        debug_print(logger, 'end optimizer and shcedule')
        # Initialize mixed-precision training
        use_mixed_precision = cfg.DTYPE == "float16"
        amp_opt_level = 'O1' if use_mixed_precision else 'O0'
        model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)
        if distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank,
                # this should be removed if we update BatchNorm stats
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
            logger.info('end distributed')
        else:
            logger.info('not distributed, singe GPU')

        arguments = {}
        arguments["iteration"] = 0

        logger.info("***********************Step 1: over***********************")
        print('\n')


        logger.info("***********************Step 1: model  construction over***********************")
        output_dir = cfg.OUTPUT_DIR


        logger.info("***********************Step 2: load pre_train_weights***********************")
        save_to_disk = get_rank() == 0
        checkpointer = DetectronCheckpointer(
            cfg, model, optimizer, scheduler, output_dir, save_to_disk, custom_scheduler=True
        )
        # if there is certain checkpoint in output_dir, load it, else load pretrained detector
        if checkpointer.has_checkpoint():
            extra_checkpoint_data = checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT,
                                                    update_schedule=cfg.SOLVER.UPDATE_SCHEDULE_DURING_LOAD)
            arguments.update(extra_checkpoint_data)
        else:
            # load_mapping is only used when we init current model from detection model.
            checkpointer.load(cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False, load_mapping=load_mapping)
        debug_print(logger, 'end load checkpointer')

        logger.info("***********************Step 2: load pre_train_weights over***********************")
        
        logger.info("***********************Step 3: load datasets ***********************")

        cfg.SOLVER.START_ITER = arguments["iteration"]
        train_data_loader = make_data_loader(
            cfg,
            mode='train',
            is_distributed=distributed,
            start_iter=arguments["iteration"],
        )
        val_data_loaders = make_data_loader(
            cfg,
            mode='test',
            is_distributed=distributed,
        )

        
        
        debug_print(logger, 'end dataloader')
        checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
        logger.info("***********************Step 3: load datasets over***********************")
        
        logger.info("Start training")
        meters = MetricLogger(delimiter="  ")
        max_iter = len(train_data_loader)
        start_iter = arguments["iteration"]
        start_training_time = time.time()
        end = time.time()

    elif "OBB" in cfg.Type:

        ##### mmcv
        args_mmcv = parse_args_OBB(mmcf,mmwei)
        cfg_mmcv = Config.fromfile(args_mmcv.config)
        if args_mmcv.cfg_options is not None:
            cfg_mmcv.merge_from_dict(args_mmcv.cfg_options)

        print('\n')
        logger.info("***********************Step 1: model  construction***********************")
        logger.info('RS construction for faster rcnn -- rbox')

        #### 加入原始cfg
        cfg_mmcv.model["ori_cfg"] = cfg
        model_mmcv = build_detector(
            cfg_mmcv.model,
            train_cfg=cfg_mmcv.get('train_cfg'),
            test_cfg=cfg_mmcv.get('test_cfg'))
        # model_mmcv.init_weights()
        logger.info('end model construction --- RS construction for faster rcnn -- rbox')

        device = cfg.MODEL.DEVICE # to  GPU 
        model_mmcv.to(device)
 
        arguments = {}
        arguments["iteration"] = 0
        cfg["mmcv"] = cfg_mmcv.data.test
        test_data_loader = make_data_loader(
        cfg,
        mode='test',
        is_distributed=distributed,
        start_iter=arguments["iteration"],
        )

       
    
        ###
        if cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "IMPPredictor":
            slow_heads = ["roi_heads.relation.box_feature_extractor",
                        "roi_heads.relation.union_feature_extractor.feature_extractor",]
        else:
            slow_heads = []

        num_batch = cfg.SOLVER.IMS_PER_BATCH
        optimizer = make_optimizer(cfg, model_mmcv, logger, slow_heads=slow_heads, slow_ratio=10.0, rl_factor=float(num_batch)) 
        ####
        # optimizer = build_optimizer(model_mmcv, cfg_mmcv.optimizer)
        scheduler = make_lr_scheduler(cfg, optimizer, logger)

        logger.info("***********************Step 1: model  construction over and load pretrained weights ***********************")
        print('\n')
        logger.info("***********************Step 2: load datasets ***********************")
        arguments = {}
        arguments["iteration"] = 0

        cfg_trian = copy.deepcopy(cfg)
        cfg_val = copy.deepcopy(cfg)
        
        cfg_trian ["mmcv"] = cfg_mmcv.data.train
        cfg_val ["mmcv"] = cfg_mmcv.data.test
        train_data_loader = make_data_loader(
             cfg_trian,
            mode='train',
            is_distributed=distributed,
            start_iter=arguments["iteration"],
        )

        val_data_loaders = make_data_loader(
            cfg_val,
            mode='test',
            is_distributed=distributed,
        )

        logger.info("***********************Step 2: load datasets over ***********************")
        print('\n')
        logger.info("***********************Step 3: Start training ***********************")
     

        ###### 读取预训练权重
        checkpoint = load_checkpoint(model_mmcv, args_mmcv.checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model_mmcv.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model_mmcv.CLASSES = dataset.CLASSES
        logger.info(args_mmcv.checkpoint)
        #### 
        model = model_mmcv
        use_mixed_precision = cfg.DTYPE == "float16"
        amp_opt_level = 'O1' if use_mixed_precision else 'O0'
        model, optRimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

        eval_modules = (model.neck, model.backbone, model.rpn_head, model.roi_head )
    
    elif "HBB" in cfg.Type:

        args_mmcv = parse_args_HBB(mmcf,mmwei)
        cfg_mmcv = Config.fromfile(args_mmcv.config)
        if args_mmcv.cfg_options is not None:
            cfg_mmcv.merge_from_dict(args_mmcv.cfg_options)

        print('\n')
        logger.info("***********************Step 1: model  construction***********************")
        logger.info('RS construction for faster rcnn -- rbox')

        #### 加入原始cfg
        cfg_mmcv.model["init_cfg"] = cfg
        model_mmcv = b_det(
            cfg_mmcv.model,
            train_cfg=cfg_mmcv.get('train_cfg'),
            test_cfg=cfg_mmcv.get('test_cfg'))
        logger.info('end model construction --- RS construction for faster rcnn -- rbox')

        device = cfg.MODEL.DEVICE # to  GPU 
        model_mmcv.to(device)
 
        arguments = {}
        arguments["iteration"] = 0
        cfg["mmcv"] = cfg_mmcv.data.test
        test_data_loader = make_data_loader(
        cfg,
        mode='test',
        is_distributed=distributed,
        start_iter=arguments["iteration"],
        )


        ###
        num_batch = cfg.SOLVER.IMS_PER_BATCH
        optimizer = make_optimizer(cfg, model_mmcv, logger, slow_heads=[], slow_ratio=10.0, rl_factor=float(num_batch)) 
        ####
        # optimizer = build_optimizer(model_mmcv, cfg_mmcv.optimizer)
        scheduler = make_lr_scheduler(cfg, optimizer, logger)

        logger.info("***********************Step 1: model  construction over and load pretrained weights ***********************")
        print('\n')
        logger.info("***********************Step 2: load datasets ***********************")
        arguments = {}
        arguments["iteration"] = 0

        cfg_trian = copy.deepcopy(cfg)
        cfg_val = copy.deepcopy(cfg)
        
        cfg_trian ["mmcv"] = cfg_mmcv.data.train
        cfg_val ["mmcv"] = cfg_mmcv.data.test
        train_data_loader = make_data_loader(
             cfg_trian,
            mode='train',
            is_distributed=distributed,
            start_iter=arguments["iteration"],
        )

        val_data_loaders = make_data_loader(
            cfg_val,
            mode='test',
            is_distributed=distributed,
        )

        logger.info("***********************Step 2: load datasets over ***********************")
        print('\n')
        logger.info("***********************Step 3: Start training ***********************")
     
        meters = MetricLogger(delimiter="  ")
        max_iter = len(train_data_loader)

        start_iter = arguments["iteration"]
        # cfg.SOLVER.START_ITER = arguments["iteration"]
        start_training_time = time.time()
        end = time.time()



        ###### 读取预训练权重
        checkpoint = load_checkpoint(model_mmcv, args_mmcv.checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model_mmcv.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model_mmcv.CLASSES = dataset.CLASSES
        logger.info(args_mmcv.checkpoint)
        #### 
        model = model_mmcv
        use_mixed_precision = cfg.DTYPE == "float16"
        amp_opt_level = 'O1' if use_mixed_precision else 'O0'
        model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

        eval_modules = (model.neck, model.backbone, model.rpn_head, model.roi_head)

        #val_result = run_val(cfg, model, val_data_loaders, distributed, logger)

    print_first_grad = True
    


    torch.cuda.empty_cache()
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations",)
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder

    data_loaders_val = make_data_loader(cfg, mode='extract', is_distributed=distributed)
    dataset = data_loaders_val[0].dataset
    print('dataset len', len(dataset))

    model.eval()
    name = 'motif'
    if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL == True:
        mode = 'predcls'
    else:
        mode = 'sgcls'
    if not os.path.exists(mix_up_path + '/' + '{}_{}_feature_dict_{}.npy'.format(mode, cfg.EXTRACT_GROUP, name)):
        results_dict = {}
        for _, batch in enumerate(tqdm(data_loaders_val[0])):
            with torch.no_grad():
                images, targets, image_ids, _, _ = batch
                # images, targets, image_ids,imgs,tar1 = batch
                targets = [target.to(device) for target in targets]
                tail_dict = model(images.to(device), targets)

        results_dict = tail_dict

        np.save(mix_up_path + '/' + '{}_{}_feature_dict_{}.npy'.format(mode, cfg.EXTRACT_GROUP, name), results_dict)

def fix_eval_modules(eval_modules):
    for module in eval_modules:
        for _, param in module.named_parameters():
            param.requires_grad = False

def main():
    parser = argparse.ArgumentParser(description="PyTorch Relation Detection Training")
    parser.add_argument(
        "--config-file",
        default="/media/dell/data1/WTZ/SGG_Frame/configs/e2e_relation_X_101_32_8_FPN_1x_trans__base.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--mm_config",
        default='/media/dell/data1/WTZ/SGG_Frame/checkpoints/1219/oriented_rcnn_swin_large_fpn_1x_dota_le90_IMP22k.py',
        help="Modify config options using the command-line",
        type=str,
    )

    parser.add_argument(
        "--mm_weight",
        default='/media/dell/data1/WTZ/SGG_Frame/PRE/OBB_large_mul/SO/epoch_12.pth',
        help="Modify config options using the command-line",
        type=str,

    )

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
    # YACS
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)
    generate_aug_results(cfg, args.local_rank, args.distributed, logger, cfg.MIXUP.FEAT_PATH, mmcf =  args.mm_config ,mmwei =  args.mm_weight)

if __name__ == "__main__":
    main()