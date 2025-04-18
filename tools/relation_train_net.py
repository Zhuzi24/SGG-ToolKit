# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Basic training script for PyTorch
"""
import copy
import argparse
import datetime
import os
import time
import os.path as osp
import torch
from mmcv.runner.checkpoint import save_checkpoint
import warnings
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.checkpoint import clip_grad_norm
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, all_gather
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip
from maskrcnn_benchmark.utils.logger import setup_logger, debug_print
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from utils import show_params_status
from tqdm import tqdm

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


def parse_args_OBB(mmcf = None,mmwei = None):

    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--work-dir', default='/SGG_ToolKit/mmrote_RS/out', help='the dir to save logs and models')

    parser.add_argument('--config', default=mmcf, help='train config file path')
    parser.add_argument('--checkpoint',default= mmwei, help='checkpoint file')

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

    parser.add_argument('--out',default='/SGG_ToolKit/mmrote_RS/checkpoints/1219/oriented_rcnn/outshiyan.pkl', help='output result file in pickle format')
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
        '--show-dir', default='/SGG_ToolKit/mmrote_RS/checkpoints', help='directory where painted images will be saved')
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

    parser.add_argument('--config',default = mmcf, help='train config file path')

    parser.add_argument('--checkpoint',default= mmwei , help='')

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

    parser.add_argument('--out',default="/SGG_ToolKit/mmdetection_RS/RSLEAP_HBB/15.pkl", help='output result file in pickle format')
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

def train(cfg, local_rank, distributed, logger, debug=False,use_GAN = False,mmcf = None,mmwei = None):


    logger.info("***********************TYPE***********************")
    logger.info("do  " + cfg.Type)

    if cfg.Type != "CV": 
        logger.info("config: " + mmcf)
        logger.info("prewei: " + mmwei)

    if cfg.Type == "CV":  # ori SGG

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

        test_data_loaders = make_data_loader(
            cfg_val,
            mode='test',
            is_distributed=distributed,
        )

        val_data_loaders = make_data_loader(
            cfg_val,
            mode='val',
            is_distributed=distributed,
        )

        logger.info("***********************Step 2: load datasets over ***********************")
        print('\n')
        logger.info("***********************Step 3: Start training ***********************")
     
        meters = MetricLogger(delimiter="  ")
        max_iter = len(train_data_loader)

        start_iter = arguments["iteration"]
        cfg.SOLVER.START_ITER = arguments["iteration"]
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
        cfg_mmcv.model["ori_cfg"] = cfg
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

        test_data_loaders = make_data_loader(
            cfg_val,
            mode='test',
            is_distributed=distributed,
        )

        val_data_loaders = make_data_loader(
            cfg_val,
            mode='val',
            is_distributed=distributed,
        )

        logger.info("***********************Step 2: load datasets over ***********************")
        print('\n')
        logger.info("***********************Step 3: Start training ***********************")
     
        meters = MetricLogger(delimiter="  ")
        max_iter = len(train_data_loader)

        start_iter = arguments["iteration"]
        cfg.SOLVER.START_ITER = arguments["iteration"]
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

      
    print_first_grad = True

    if cfg.Only_val:
        val_result = run_val(cfg, model, val_data_loaders, distributed, logger, output_folder = cfg.val_outpath)
        sys.exit() 

    if cfg.Only_test:
        val_result = run_val(cfg, model, test_data_loaders, distributed, logger, output_folder = cfg.test_outpath)
        sys.exit() 



    for iteration, (images, targets, _ , imgs, tar1) in enumerate(train_data_loader, start_iter):  
        
        if any(len(target) < 1 for target in targets):
             logger.error(
            f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}")
        data_time = time.time() - end
        iteration = iteration + 1 + cfg.ite_resume 



        arguments["iteration"] = iteration


        model.train()
        # model.eval()self.model.embedding.weight.data
        fix_eval_modules(eval_modules) # 先注释

        images = images.to(device)
        targets = [target.to(device) for target in targets]
        
        
        loss_dict = model(images, targets, ite=iteration, logger = logger, sgd_data = [imgs, tar1] if imgs is not None else None)

        losses = sum(loss for loss in loss_dict.values())
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)
        
        optimizer.zero_grad()
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()
        verbose = (iteration % cfg.SOLVER.PRINT_GRAD_FREQ) == 0 or print_first_grad  # print grad or not


        print_first_grad = False

        clip_grad_norm([(n, p) for n, p in model.named_parameters() if p.requires_grad],
                       max_norm=cfg.SOLVER.GRAD_NORM_CLIP, logger=logger, verbose=verbose, clip=True)


        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % cfg.Print_iter == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[-1]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        
        if iteration % cfg.SOLVER.CHECKPOINT_PERIOD == 0 or iteration == max_iter:
            if cfg.Type != "CV":
                filename =  cfg.OUTPUT_DIR + "/" + str(iteration)+ ".pth" 
                meta = {}
                meta["CLASSES"]  = ('ship','boat','crane','goods_yard','tank','storehouse','breakwater','dock','airplane','boarding_bridge','runway','taxiway','terminal','apron','gas_station','truck','car','truck_parking','car_parking','bridge','cooling_tower','chimney','vapor','smoke','genset','coal_yard','lattice_tower', 'substation', 'wind_mill','cement_concrete_pavement', 'toll_gate', 'flood_dam', 'gravity_dam', 'ship_lock','ground_track_field','basketball_court','engineering_vehicle', 'foundation_pit', 'intersection', 'soccer_ball_field','tennis_court','tower_crane','unfinished_building','arch_dam','roundabout','baseball_diamond','stadium','containment_vessel')  

                save_checkpoint(model, filename, optimizer=optimizer, meta=meta)
            else:
                        
                checkpointer.save("model_{:07d}".format(iteration), **arguments)
                if iteration == max_iter:
                    checkpointer.save("model_final", **arguments)
        
        val_result = None 
        if iteration % cfg.SOLVER.VAL_PERIOD == 0:  # 
             val_result = run_val(cfg, model, val_data_loaders, distributed, logger,output_folder = cfg.outpath)
     

        if cfg.SOLVER.SCHEDULE.TYPE == "WarmupReduceLROnPlateau":
            scheduler.step(val_result, epoch=iteration)
            if scheduler.stage_count >= cfg.SOLVER.SCHEDULE.MAX_DECAY_STEP:
                logger.info("Trigger MAX_DECAY_STEP at iteration {}.".format(iteration))
                checkpointer.save("model_final", **arguments)
                break
        else:
            scheduler.step()
        torch.cuda.empty_cache() 

        if iteration == max_iter:
            break


    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    return model
 
        


def fix_eval_modules(eval_modules):
    for module in eval_modules:
        for _, param in module.named_parameters():
            param.requires_grad = False
        # DO NOT use module.eval(),
        # otherwise the module will be in the test mode,
        # i.e., all self.training condition is set to False


def run_val(cfg, model, val_data_loaders, distributed, logger,m = None,ite = None,CCM = None,output_folder = None,vae = None):
    val = 1
    if distributed:
        model = model.module
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

    dataset_names = cfg.DATASETS.VAL
    val_result = []
    for dataset_name, val_data_loader in zip(dataset_names, val_data_loaders):
        dataset_result = inference(
            cfg,
            model,
            val_data_loader,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            logger=logger,
            m=m,
            val=val,
            ite=ite,
            CCM = CCM,
            vae = vae
        )
        synchronize()
        val_result.append(dataset_result)
    # support for multi gpu distributed testing
    gathered_result = all_gather(torch.tensor(dataset_result).cpu())
    gathered_result = [t.view(-1) for t in gathered_result]
    gathered_result = torch.cat(gathered_result, dim=-1).view(-1)
    valid_result = gathered_result[gathered_result >= 0]
    val_result = float(valid_result.mean())
    del gathered_result, valid_result
    torch.cuda.empty_cache()
    return val_result


def run_test(cfg, model, distributed, logger, m = None,CCM = None):
    val = 2
    if distributed:
        model = model.module
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
    data_loaders_val = make_data_loader(cfg, mode='test', is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            cfg,
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            logger=logger,
            val=val,
        )
        synchronize()


def main(debug=False):  
 

    parser = argparse.ArgumentParser(description="PyTorch Relation Detection Training")
    parser.add_argument(
        "--config-file",
        default='/SGG_ToolKit/configs/e2e_relation_X_101_32_8_FPN_1x_trans__base.yaml',
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    parser.add_argument("--local_rank", default=0)

    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )

    parser.add_argument(
        "--log_name",
        default="log.txt",
        help="Do not test the final model",
        type=str,
    )

    parser.add_argument(
        "--mm_config",
        default='configs/RSOBB/STAR_obb_predcls_sgcls.py',
        help="Modify config options using the command-line",
        type=str,
    )

    parser.add_argument(
        "--mm_weight",
        default='PRE_WEI/OBB_Swin.pth',
        help="Modify config options using the command-line",
        type=str,

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
    local_rank = int(os.environ['LOCAL_RANK']) if "WORLD_SIZE" in os.environ else 0
    if args.distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)


    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank(),filename=args.log_name)
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

    
    model = train(cfg, local_rank, args.distributed, logger, debug=debug, mmcf =  args.mm_config ,mmwei =  args.mm_weight)
    if not args.skip_test:
        run_test(cfg, model, args.distributed, logger)



if __name__ == "__main__":
    import sys

    print('running with system paths :', sys.path)
    main()

    


    