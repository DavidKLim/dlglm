# dlglm
Package for Deeply-Learned Generalized Linear Model (dlglm). Uses reticulated Python to run deep learning models through R. 

## Installation
In order to install the released version of the `dlglm` R package, it is important to correctly install Python, and the required Python modules. The package was tested and developed using Python3, and the versions of the Python modules listed below. `dlglm` may work with a different version of Python and/or different versions of the listed modules, but may yield unexpected results. Below is a guide on how to replicate the tested installation of `dlglm` from scratch. Disclaimer: in order to run `dlglm`, you will need a CUDA-enabled graphics card. Please ensure that you have a CUDA-enabled GPU on your system, and check the version of CUDA that is supported by your GPU card before your installation!

First, you want to make sure you have a working version of Python3 on your system. `dlglm` was tested on Python versions 3.6.6 and 3.7.6.

Next, you want to install the necessary Python module dependencies. Below is a list of Python modules to install.\
numpy=1.19.0\
scipy=1.4.1\
h5py=2.10.0\
pandas=0.25.2\
matplotlib=3.1.3\
cloudpickle=1.2.2\
torch=1.4.0\
torchvision=0.5.0\
tensorflow = 1.14.0\
tensorflow-probability=0.7.0\
tensorboard=1.14.0\
\
The `numpy` and `cloudpickle` packages can be installed using a either the pip or conda installer. One may prefer to use conda if the aim is to create a virtual environment to test out `dlglm` without affecting the installation of these dependencies on the entire system. Please refer [here](https://pip.pypa.io/en/stable/installation/) for how to install the pip installer, and [here](https://docs.anaconda.com/anaconda/install/index.html) for how to install Anaconda and the conda installer. Assuming you are using the pip installer, these packages can be installed by opening your command line, and inputting: ``` pip3 install numpy==1.19.0 cloudpickle==1.2.2 scipy==1.4.1 h5py==2.10.0```.\
\
Before installing the Pytorch modules, you should first confirm that you have a working NVIDIA CUDA installation on your system. If you have not configured a CUDA installation for your system, please follow the guide [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions) and follow the steps to install the CUDA toolkit and drivers on your system. Then, the `torch` and `torchvision` modules can be installed by following the instructions listed [here](https://pytorch.org/get-started/previous-versions/). To install the listed versions of these modules, input the following into the command line: ```pip3 install torch==1.4.0 torchvision==0.5.0```. To check that the installation was successfully completed, open a Python3 console using the command `python3` and type:
```
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```
This will verify that the correct version of Pytorch has been installed, and that a CUDA-enabled GPU is available for use in Pytorch.

After making sure your Python environment is properly set-up, you can install the `dlglm` R package by using the command: `devtools::install_github("DavidKLim/dlglm")`.

## Minimal Working Example

Below is a minimal working example to verify that your `dlglm` installation was completed:

``` r
library(MASS)
library(dlglm)
beta = c(-0.25,0.25)
e = rnorm(0,0.1)
n = 10000
p = 2
X = MASS::mvrnorm(n=n, mu=rep(0,p), Sigma=diag(p))
Y = X %*% beta + e
mask_x = matrix(rbinom(n*p,1,0.5),nrow=n,ncol=p)
mask_y = matrix(rep(1,n), ncol=1)
data_types_x = rep("real",p)

family="Gaussian"
link="identity"
g = c(rep("train",8000), rep("valid",1000), rep("test",1000))
res = dlglm::dlglm(dir_name=".", X=X, Y=Y, mask_x=mask_x, mask_y=mask_y, g=g, covars_r_x=rep(1,p), covars_r_y=1, learn_r=T, data_types_x=data_types_x, Ignorable=F, family=family, link=link, normalize=F, early_stop=T, trace=F)
str(res)
```

NOTE: `dlglm` is built using a Python backend, using the `reticulate` R package. You may be prompted to install Miniconda when running `dlglm` for the first time. This message is from the `reticulate` package, and we advise that you decline the installation and use the system Python version as installed above. Otherwise, Python module dependencies listed above will need to be installed again for the Miniconda version of Python, as `reticulate` will default to the new installation of Python.
