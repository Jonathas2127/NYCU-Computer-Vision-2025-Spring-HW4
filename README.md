# NYCU-Computer-Vision-2025-Spring-HW4
student ID: 111550169

Name: 陳宣澔

## Introduction
This project implements PromptIR, a unified transformer-based framework for blind image restoration. PromptIR dynamically adapts to various degradations—such as rain, snow, noise, and haze—using learned degradation-aware prompts. Apply the model to a custom dataset for restoring clean images from their degraded counterparts, leveraging PyTorch Lightning for efficient training and checkpointing. Key features include automatic image pairing, multi-scale transformer blocks, and prompt-guided attention modules for flexible and robust restoration.

## How to install
I run kaggle.py on Kaggle Notebooks.
Upload datasets and model.py, preprocess.py, utils.py as models to Kaggle first and open a new notebook.

### First cell:

    !git clone https://github.com/va1shn9v/PromptIR.git
    !pip install einops lightning
    !pip install timm

### Second cell:

    import sys
    sys.path.append('/kaggle/input/a/other/default/1') # models

### Third cell:

kaggle.py

### Fourth cell:

    !cp /kaggle/input/a/other/default/1/*.py /kaggle/working/
    !python train.py --de_type derain --epochs 75

Run all cells and download the model path in output.

Then, I run inference.py on PC with the model path downloading from the training section on Kaggle.

## Performance snapshot
![image](https://github.com/Jonathas2127/pictures/blob/main/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%202025-05-27%20185154.png)
