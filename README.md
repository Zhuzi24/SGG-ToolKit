[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/Zhuzi24/SGG-ToolKit/graphs/commit-activity)
[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/Zhuzi24/SGG-ToolKit)
<img alt="GitHub watchers" src="https://img.shields.io/github/watchers/Zhuzi24/SGG-ToolKit?style=social"> <img alt="GitHub stars" src="https://img.shields.io/github/stars/Zhuzi24/SGG-ToolKit?style=social"> <img alt="GitHub forks" src="https://img.shields.io/github/forks/WTZ6666/Video-Dynamic-Scene-Graph-Generation?style=social">

# Scene Graph Generation in Large-Size VHR Satellite Imagery: A Large-Scale Dataset and A Context-Aware Approach

The official implementation of the paper "[Scene Graph Generation in Large-Size VHR Satellite Imagery: A Large-Scale Dataset and A Context-Aware Approach](https://arxiv.org/abs/)".

## 📢 Latest Updates
:fire::fire::fire: Last Updated on 2024.06.13 :fire::fire::fire:

- **2024.06.13**: Update project.

## ⭐️ Highlights

We construct RSG, the first large-scale dataset for scene graph generation in large-size VHR SAI. Containing  more than  `210,000` objects and over `400,000` triplets for SGG in large-size VHR SAI.

<p align="center">
<img src="demo/distr.jpg" alt="scatter" width="98%"/> 
</p>

https://private-user-images.githubusercontent.com/29257168/339049597-2d027f2c-8911-45ba-b4dd-7f95111465a9.mp4


## 📌 Abstract

Scene graph generation (SGG) in satellite imagery (SAI) benefits promoting intelligent understanding of geospatial scenarios from perception to cognition. In SAI, objects exhibit great variations in scales and aspect ratios, and there exist rich relationships between objects (even between spatially disjoint objects), which makes it necessary to holistically conduct SGG in large-size very-high-resolution (VHR) SAI. However, the lack of SGG datasets with large-size VHR SAI has constrained the advancement of SGG in SAI. Due to the complexity of large-size VHR SAI, mining triplets <subject, relationship, object$> in large-size VHR SAI heavily relies on long-range contextual reasoning. Consequently, SGG models designed for small-size natural imagery are not directly applicable to large-size VHR SAI. To address the scarcity of datasets, this paper constructs a large-scale dataset for SGG in large-size VHR SAI with image sizes ranging from 512 × 768 to 27,860 × 31,096 pixels, named RSG, encompassing over 210,000 objects and more than 400,000 triplets. To realize SGG in large-size VHR SAI, we propose a context-aware cascade cognition (CAC) framework to understand SAI at three levels: object detection (OBD), pair pruning and relationship prediction. As a fundamental prerequisite for SGG in large-size SAI, a holistic multi-class object detection network (HOD-Net) that can flexibly integrate multi-scale contexts is proposed. With the consideration that there exist a huge amount of object pairs in large-size SAI but only a minority of object pairs contain meaningful relationships, we design a pair proposal generation (PPG) network via adversarial reconstruction to select high-value pairs. Furthermore, a relationship prediction network with context-aware messaging (RPCM) is proposed to predict the relationship types of these pairs. To promote the development of SGG in large-size VHR SAI, this paper releases a SAI-oriented SGG toolkit with about 30 OBD methods and 10 SGG methods, and develops a benchmark based on RSG where our HOD-Net and RPCM significantly outperform the state-of-the-art methods in both OBD and SGG tasks. **The RSG dataset will be made publicly available at [RSG](https://linlin-dev.github.io/project/RSG.html)**.



## 🛠️ Installation
Check [INSTALL.md](INSTALL.md) for installation instructions.

## 🚀 Dataset
Check [DATASET.md](DATASET.md) for instructions of dataset preprocessing.

## 🚀 Metrics and Results
Explanation of metrics in our toolkit and reported results for OBD and SGG are given in [METRICS.md](METRICS.md). If you are only involved in OBB/HBB object detection, you can refer to [RSG-MMRotate](https://github.com/yangxue0827/RSG-MMRotate) and [RSG-MMDetection](https://github.com/Zhuzi24/RSG-MMDetection).

<!-- ### Oriented Object Detection

|  Detector  | mAP | Configs | Download | Note |
| :--------: |:---:|:-------:|:--------:|:----:|
| Deformable DETR | 17.1 | [deformable_detr_r50_1x_rsg](configs/ars_detr/deformable_detr_r50_1x_rsg.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/deformable_detr_r50_1x_rsg.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/deformable_detr_r50_1x_rsg-fe862bb3.pth?download=true) |
| ARS-DETR | 28.1 | [dn_arw_arm_arcsl_rdetr_r50_1x_rsg](configs/ars_detr/dn_arw_arm_arcsl_rdetr_r50_1x_rsg.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/dn_arw_arm_arcsl_rdetr_r50_1x_rsg.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/dn_arw_arm_arcsl_rdetr_r50_1x_rsg-cbb34897.pth?download=true) |
| RetinaNet | 21.8 | [rotated_retinanet_hbb_r50_fpn_1x_rsg_oc](configs/rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_rsg_oc.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/rotated_retinanet_hbb_r50_fpn_1x_rsg_oc.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/rotated_retinanet_hbb_r50_fpn_1x_rsg_oc-3ec35d77.pth?download=true) |
| ATSS | 20.4 | [rotated_atss_hbb_r50_fpn_1x_rsg_oc](configs/rotated_atss/rotated_atss_hbb_r50_fpn_1x_rsg_oc.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/rotated_atss_hbb_r50_fpn_1x_rsg_oc.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/rotated_atss_hbb_r50_fpn_1x_rsg_oc-f65f07c2.pth?download=true) | 
|  KLD  |  25.0  |   [rotated_retinanet_hbb_kld_r50_fpn_1x_rsg_oc](configs/kld/rotated_retinanet_hbb_kld_r50_fpn_1x_rsg_oc.py)  |  [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/rotated_retinanet_hbb_kld_r50_fpn_1x_rsg_oc.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/rotated_retinanet_hbb_kld_r50_fpn_1x_rsg_oc-343a0b83.pth?download=true) |
|  GWD  |  25.3  |   [rotated_retinanet_hbb_gwd_r50_fpn_1x_rsg_oc](configs/gwd/rotated_retinanet_hbb_gwd_r50_fpn_1x_rsg_oc.py)  |  [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/rotated_retinanet_hbb_gwd_r50_fpn_1x_rsg_oc.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/rotated_retinanet_hbb_gwd_r50_fpn_1x_rsg_oc-566d2398.pth?download=true) |
| KFIoU |  25.5  |   [rotated_retinanet_hbb_kfiou_r50_fpn_1x_rsg_oc](configs/kfiou/rotated_retinanet_hbb_kfiou_r50_fpn_1x_rsg_oc.py)  |  [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/rotated_retinanet_hbb_kfiou_r50_fpn_1x_rsg_oc.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/rotated_retinanet_hbb_kfiou_r50_fpn_1x_rsg_oc-198081a6.pth?download=true) |
| S2A-Net | 27.3 | [s2anet_r50_fpn_1x_rsg_le135](configs/s2anet/s2anet_r50_fpn_1x_rsg_le135.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/s2anet_r50_fpn_1x_rsg_le135.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/s2anet_r50_fpn_1x_rsg_le135-42887a81.pth?download=true) |
| FCOS  |  28.1  |   [rotated_fcos_r50_fpn_1x_rsg_le90](configs/rotated_fcos/rotated_fcos_r50_fpn_1x_rsg_le90.py)  |  [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/rotated_fcos_r50_fpn_1x_rsg_le90.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/rotated_fcos_r50_fpn_1x_rsg_le90-a579fbf7.pth?download=true) | 
| CSL | 27.4 | [rotated_fcos_csl_gaussian_r50_fpn_1x_rsg_le90](configs/rotated_fcos/rotated_fcos_csl_gaussian_r50_fpn_1x_rsg_le90.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/rotated_fcos_csl_gaussian_r50_fpn_1x_rsg_le90.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/rotated_fcos_csl_gaussian_r50_fpn_1x_rsg_le90-6ab9a42a.pth?download=true) | 
| RepPoints  | 19.7 | [rotated_reppoints_r50_fpn_1x_rsg_oc](configs/rotated_reppoints/rotated_reppoints_r50_fpn_1x_rsg_oc.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/rotated_reppoints_r50_fpn_1x_rsg_oc.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/rotated_reppoints_r50_fpn_1x_rsg_oc-7a6c59b9.pth?download=true) |
| CFA | 25.1 | [cfa_r50_fpn_1x_rsg_le135](configs/cfa/cfa_r50_fpn_1x_rsg_le135.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/cfa_r50_fpn_1x_rsg_le135.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/cfa_r50_fpn_1x_rsg_le135-287f6b84.pth?download=true) |
| Oriented RepPoints  |  27.0  |   [oriented_reppoints_r50_fpn_1x_rsg_le135](configs/oriented_reppoints/oriented_reppoints_r50_fpn_1x_rsg_le135.py)  |  [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/oriented_reppoints_r50_fpn_1x_rsg_le135.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/oriented_reppoints_r50_fpn_1x_rsg_le135-06389ea6.pth?download=true) | |
| SASM  |  28.2  |   [sasm_reppoints_r50_fpn_1x_rsg_oc](configs/sasm_reppoints/sasm_reppoints_r50_fpn_1x_rsg_oc.py)  |  [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/sasm_reppoints_r50_fpn_1x_rsg_oc.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/sasm_reppoints_r50_fpn_1x_rsg_oc-4f1ca558.pth?download=true) | [p_bs=2](https://github.com/yangxue0827/RSG-MMRotate/blob/05c0064cbcd5c44437321b50e1d2d4ee9b4445db/mmrotate/models/detectors/single_stage_crop.py#L310) |
| Faster RCNN | 32.6 | [rotated_faster_rcnn_r50_fpn_1x_rsg_le90](configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_rsg_le90.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/rotated_faster_rcnn_r50_fpn_1x_rsg_le90.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/rotated_faster_rcnn_r50_fpn_1x_rsg_le90-9a832bc2.pth?download=true) |
| Gliding Vertex | 30.7 | [gliding_vertex_r50_fpn_1x_rsg_le90](configs/gliding_vertex/gliding_vertex_r50_fpn_1x_rsg_le90.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/gliding_vertex_r50_fpn_1x_rsg_le90.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/gliding_vertex_r50_fpn_1x_rsg_le90-5c0bc879.pth?download=true) |
| Oriented RCNN | 33.2 | [oriented_rcnn_r50_fpn_1x_rsg_le90](configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_rsg_le90.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/oriented_rcnn_r50_fpn_1x_rsg_le90.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/oriented_rcnn_r50_fpn_1x_rsg_le90-0b66f6a4.pth?download=true) |
| RoI Transformer | 35.7 | [roi_trans_r50_fpn_1x_rsg_le90](configs/roi_trans/roi_trans_r50_fpn_1x_rsg_le90.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/roi_trans_r50_fpn_1x_rsg_le90.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/roi_trans_r50_fpn_1x_rsg_le90-e42f64d6.pth?download=true) |
| ReDet | 39.1 | [redet_re50_refpn_1x_rsg_le90](configs/redet/redet_re50_refpn_1x_rsg_le90.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/redet_re50_refpn_1x_rsg_le90.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/redet_re50_refpn_1x_rsg_le90-d163f450.pth?download=true) | [ReResNet50](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/re_resnet50_c8_batch256-25b16846.pth?download=true) |
| Oriented RCNN | 40.7 | [oriented_rcnn_swin-l_fpn_1x_rsg_le90](configs/oriented_rcnn/oriented_rcnn_swin-l_fpn_1x_rsg_le90.py) | [log](https://huggingface.co/yangxue/RSG-MMRotate/raw/main/oriented_rcnn_swin-l_fpn_1x_rsg_le90.log) \| [ckpt](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/oriented_rcnn_swin-l_fpn_1x_rsg_le90-fe6f9e2d.pth?download=true) | [Swin-L](https://huggingface.co/yangxue/RSG-MMRotate/resolve/main/swin_large_patch4_window7_224_22k_20220412-aeecf2aa.pth?download=true) | -->

## 🖊️ Citation

If you find this work helpful for your research, please consider giving this repo a star ⭐ and citing our paper:

```bibtex
@article{li2024scene,
  title={Scene Graph Generation in Large-Size VHR Satellite Imagery: A Large-Scale Dataset and A Context-Aware Approach},
  author={L1, Yansheng and Wang, Linlin and Wang, Tingzhu and Yang, Xue and Wang, Qi and Sun, Xian and Wang, Wenbin and Luo, Junwei and Deng, Youming and Li, Haifeng and Dang, Bo and Zhang, Yongjun and Yan Junchi},
  journal={arXiv preprint arXiv:},
  year={2024}
}
```
## Acknowledgment
Our code is based on [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), [MMDetection](https://github.com/open-mmlab/mmdetection) and [MMRotate](https://github.com/open-mmlab/mmrotate), we sincerely thank them.
<!-- ## 📃 License

This project is released under the [Apache license](LICENSE). Parts of this project contain code and models from other sources, which are subject to their respective licenses. -->
