# ResNetScratch
## Introduction
This project implements a convolutional neural network from scratch using PyTorch to classify images from the CIFAR-10 dataset. It reproduces the ResNet-18 architecture using custom residual blocks (named Brick) and evaluates its performance through training, validation, and testing phases.

## Features
- Custom implementation of ResNet-18 using residual connections
- CIFAR-10 dataset loading, splitting, and normalization
- Training pipeline with model checkpointing
- Validation and test accuracy reporting
- GPU support for accelerated training

## Dataset
CIFAR-10 is a dataset consisting of 60,000 32x32 color images in 10 classes, with 6,000 images per class. This implementation downloads it automatically and splits it into:
- Training set: 40,000 images
- Validation set: 10,000 images
- Test set: 10,000 images

## Implementation Details
Model Architecture:
Custom implementation of ResNet-18 built using a reusable Brick class for residual blocks. It includes four residual layers (l1 to l4), global average pooling, and a final fully connected layer.  
Training:
- Optimizer: SGD with momentum
- Loss function: CrossEntropyLoss
- 50 training epochs

Evaluation:
- After training, the best model is evaluated on the test dataset.
- Final test accuracy is printed at the end.

## Dependencies
- Python 3.x
- PyTorch
- torchvision
- CUDA-enabled GPU (optional, for acceleration)
