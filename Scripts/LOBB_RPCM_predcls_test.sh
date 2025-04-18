#!/bin/bash 

export CUDA_VISIBLE_DEVICES=2
export NUM_GUP=1

MODEL_NAME='LOBB_RPCM_predcls_test'
path="Checkpoints/${MODEL_NAME}/"
mkdir -p "$path"

#  "PRE_Wei/OBB_swin_L_OBD.pth" \
python3 \
  tools/relation_train_net.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x_trans_base.yaml" \
  --mm_config "configs/RSOBB/STAR_obb_predcls_sgcls.py" \
  --mm_weight "Weights/LOBB_RPCM_PredCls.pth" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
  MODEL.ROI_RELATION_HEAD.PREDICTOR RPCM \
  SOLVER.WARMUP_ITERS 500 \
  DTYPE "float32" \
  GLOVE_DIR glove\
  SOLVER.IMS_PER_BATCH 4 TEST.IMS_PER_BATCH $NUM_GUP \
  SOLVER.MAX_ITER 10000 SOLVER.BASE_LR 1e-3 \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
  SOLVER.STEPS "(6000, 8500)" SOLVER.VAL_PERIOD 2000 \
  SOLVER.CHECKPOINT_PERIOD 1000  \
  OUTPUT_DIR "$path" \
  SOLVER.PRE_VAL False \
  SOLVER.GRAD_NORM_CLIP 5.0 \
  Type "Large_RS_OBB"  \
  filter_method "PPG" \
  Only_test True \
  test_outpath "$path";