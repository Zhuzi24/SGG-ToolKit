
## instructions
<!-- Download the [STAR_SGG.zip]() file and unzip it, then set the paths in maskrcnn_benchmark/config/paths_catalog.py on line 10. \
STAR_SGG.zip contains the SGG need files, e.g. STAR-SGG-with-attri.h5, etc. -->

First download the dataset [images](https://huggingface.co/datasets/Zhuzi24/STAR) and [annotation files](), and unzip them. Note that the downloaded images are separated by TRAIN, VAL, and TEST sets, and you need to merge all of them into one file named STAR_img.

Then put the image and annotation files into the STAR_SGG/STAR_img folder (you can explicitly do this yourself) in the specific path format:
```
STAR_SGG/
├── STAR_img     # Contains all images, total 1273.
├── STAR-SGG-with-attri.h5
├── ...
├── ...
└── STAR-SGG-dicts-with-attri.json
```
Finally, setup path "yourpath/STAR_SGG/" in maskrcnn_benchmark/config/paths_catalog.py on line 10.
