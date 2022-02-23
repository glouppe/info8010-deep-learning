# Installation instructions
\
To complete the assignments, you need a working Python installation with Jupyter and the following libraries: numpy, scipy, matplotlib, 
scikit-learn, pytorch, torchvision and pillow. We recommend using Anaconda although this is not mandatory.

## Installing Anaconda
\
You can find Anaconda installation instructions in the [official Anaconda documentation](https://docs.anaconda.com/anaconda/install/index.html).

Once anaconda is installed, create a virtual environment dedicated to the homeworks using the following command
```
conda create --name deep_learning
```
You can activate the environment using the command
```
conda activate deep_learning
```
and deactivate it using
```
conda deactivate
```
## Installing pytorch
\
To install Pytorch, go to [this webpage](https://pytorch.org/get-started/locally/) and select the different boxes according to your system.
Be sure to activate your environment with `conda activate deep_learning` before entering the install command.
We recommend installing the stable version. 
If you have an NVIDIA GPU on your computer pick the version of CUDA installed on your machine, otherwise pick CPU. 
AMD GPUs cannot be used for deep learning as the libraries usually rely on CUDA, if you have an AMD GPU, pick the CPU version.

## Installing the other libraries
\
You can install all the remaining librairies using the following command
```
conda install jupyter numpy scipy matplotlib pillow scikit-learn
```
or find the installation commands in the documentation: 
[jupyter](https://anaconda.org/anaconda/jupyter),
[numpy](https://anaconda.org/anaconda/numpy),
[matplotlib](https://anaconda.org/conda-forge/matplotlib),
[scikit-learn](https://anaconda.org/anaconda/scikit-learn) and
[pillow](https://anaconda.org/anaconda/pillow)

## Running the notebook
\
To open the first homework in a notebook, run the following command
```
jupyter notebook homework1.ipynb
```
A link will be prompted in the terminal, open this link in a browser and you can start editing from there.
