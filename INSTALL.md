## Installation

Detailed environment configuration can be found at [SGG_ToolKit_environment.yml](SGG_ToolKit_environment.yml).\
We tried to configure the environment on different devices to test the code, and found that the results of the test have a very small float with the results of the paper (due to differences in devices, library versions, etc.), if you want to test to achieve the same accuracy, you need to be consistent with the version of the library in [SGG_ToolKit_environment.yml](SGG_ToolKit_environment.yml).

We have tried the environment many times and found that the torch version is fine depending on your computer configuration of other versions, but sometimes you may need to synchronise the modification of other libraries, the following is for reference only.

### Step-by-step installation

```bash
# download the SGG_ToolKit project manually or via git.
cd SGG_ToolKit

# Creating a virtual environment via conda
conda create -n SGG_ToolKit python=3.8  
# Activate the virtual environment
conda activate SGG_ToolKit

# Start configuring other libraries
conda install ipython scipy h5py
pip install ninja yacs cython matplotlib tqdm opencv-python overrides shapely ipdb

# Installing torch, after my attempts, is more successful with pip than conda!, the installation commands can be found here: https://pytorch.org/get-started/previous-versions/
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# Compile cocoapi and apex, the same as https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch
download cocoapi.zip in https://huggingface.co/Zhuzi24/STAR_OBJ_REL_WEIGHTS/tree/main, then unzip it # git clone https://github.com/cocodataset/cocoapi.git, can try multiple times or check the network and if it fails
cd cocoapi/PythonAPI
python setup.py build_ext install

download apex.zip in https://huggingface.co/Zhuzi24/STAR_OBJ_REL_WEIGHTS/tree/main, then unzip it # git clone https://github.com/NVIDIA/apex.git, can try multiple times or check the network and if it fails
cd apex
# "python setup.py install --cuda_ext --cpp_ext" will limit the specific version, designed to modify the setup.py, after trying it will work
python setup.py install 

# Some libraries to keep in mind for installation
 pip install torch-geometric==2.0.4  torch-scatter==2.0.7 torch-sparse==0.6.9 # The first installation may take a long time to compile, please be patient!  
 pip install numpy==1.23.5

# Overall compilation
python setup.py build develop

# need to install mmrotate and mmdetection for STAR dataset
cd mmrote_RS
pip install openmim
mim install mmcv-full==1.7.1 
mim install mmdet==2.26.0
pip install -r requirements/build.txt
pip install -v -e .

Important:
In order to run the HBB type of SGG, the following modifications are also required:
Use mmdet (https://github.com/Zhuzi24/SGG-ToolKit/tree/main/lib_mmdet/mmdet) to replace mmdet in the virtual environment (/xxx/miniconda3/envs/SGG_ToolKit/lib/python3.8/site-packages/mmdet).
/xxx/miniconda3/envs/SGG_Frame/lib/python3.8/site-packages/mmdet, usually found under the conda configuration file, an example of which is shown above.

### Some of the issues that may arise (will be updated continuously)
1. AssertionError: MMCV==1.7.1 is used but incompatible. Please install mmcv>=1.4.5, <=1.6.0.
 A: replacement version of "mim install mmcv==1.6.0"

2. AssertionError: Egg-link /xxx/envs/SGG_A/lib/python3.8/site-packages/mmrotate.egg-link (to /xxx/mmrote_RS) does not match installed location of mmrotate (at /xxx/SGG_ToolKit/mmrote_RS)
 A: Deletion of compiled files mmrotate.egg-info


