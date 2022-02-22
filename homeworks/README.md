# Installation instructions
To complete the assignments, you need a working python installation with jupyter and the following libraries: numpy, scipy, matplotlib, 
scikit-learn, pytorch, torchvision and pillow. We recommend using anaconda although this is not mandatory.

## Installing anaconda
You can Anaconda installation instructions here https://docs.anaconda.com/anaconda/install/index.html .

Once anaconda is installed, create a virtual environment dedicated to the homeworks using the following command
```
conda create --name deep_learning
```
You can activate the environment using the command
```
source activate deep_learning
```
and deactivate it using
```
conda deactivate
```
## Installing pytorch
To install pytorch, go to this webpage https://pytorch.org/get-started/locally/ and select the different boxes according to your system. 
We recommend installing the stable version. 
If you have an NVIDIA GPU on your computer pick the version of cuda installed on your machine, otherwise pick CPU. 
AMD GPUs cannot be used for deep learning as the libraries usually reliy on cuda, if you have an AMD GPU, pick the CPU version.

## Installing the other libraries
Head to the anaconda doc page of the library to find the installation command.
* jupyter: https://anaconda.org/anaconda/jupyter
* numpy: https://anaconda.org/anaconda/numpy
* scipy: https://anaconda.org/anaconda/scipy
* matplotlib: https://anaconda.org/conda-forge/matplotlib
* scikit-learn: https://anaconda.org/anaconda/scikit-learn
* torchvision: https://anaconda.org/pytorch/torchvision
* pillow: https://anaconda.org/anaconda/pillow

## Running the notebook
To open the first homework in a notebook, run the following command
```
jupyter notebook homework1.ipynb
```
A link will be prompted in the terminal, open this link in a browser and you can start editing from there.
