## Installation

Detailed environment configuration can be found at [SGG_ToolKit_environment.yml](SGG_ToolKit_environment.yml).

I've tried the environment many times and found that the torch version is fine depending on your computer configuration of other versions, but sometimes you may need to synchronise the modification of other libraries, the following is for reference only.

### Step-by-step installation

```bash
# download the SGG_ToolKit project manually or via git.
cd SGG_ToolKit

# Creating a virtual environment via conda
conda create --n SGG_ToolKit python=3.8  
# Activate the virtual environment
conda activate SGG_ToolKit

# Start configuring other libraries
conda install ipython scipy h5py
pip install ninja yacs cython matplotlib tqdm opencv-python overrides

# Installing torch, after my attempts, is more successful with pip than conda!, the installation commands can be found here: https://pytorch.org/get-started/previous-versions/
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# Compile cocoapi and apex, the same as https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch
git clone https://github.com/cocodataset/cocoapi.git # Can try multiple times or check the network and if it fails
cd cocoapi/PythonAPI
python setup.py build_ext install

git clone https://github.com/NVIDIA/apex.git # Can try multiple times or check the network and if it fails
cd apex
# "python setup.py install --cuda_ext --cpp_ext" will limit the specific version, designed to modify the setup.py, after trying it will work
python setup.py install 

# Some libraries to keep in mind for installation
 pip install torch-geometric==2.0.4  torch-scatter==2.0.7 torch-sparse==0.6.9
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

### Some of the issues that may arise (will be updated continuously)
1. AssertionError: MMCV==1.7.1 is used but incompatible. Please install mmcv>=1.4.5, <=1.6.0.
 A1: replacement version of "mim install mmcv==1.6.0"

