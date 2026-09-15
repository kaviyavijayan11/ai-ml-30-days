# Image Classifier - Transfer Learning

## Project Overview
Multi-class image classifier using Transfer Learning with 
MobileNetV2, classifying images into 10 CIFAR-10 categories.

## Approach
- Used pretrained MobileNetV2 (trained on ImageNet, 1.4M images)
- Froze base model weights, added custom classification head
- Only trained ~165K parameters (6.8% of total) instead of 
  training from scratch

## Debugging Journey (Key Learning)
Initial attempt used MobileNetV2 with CIFAR-10's native 32x32 
images, resulting in only 29.5% accuracy. Investigation revealed 
MobileNetV2 requires minimum 96x96 input size for its pretrained 
weights to transfer meaningfully. After resizing images to 96x96 
and rebuilding the model with correct input_shape, accuracy 
improved to 77.40% — a 47.9 percentage point gain.

## Results
- Final Test Accuracy: 77.40%
- Improvement from initial (broken) version: +47.90%
- Trainable params: 165,258 (6.8% of total 2.4M)

## Key Techniques
1. Transfer Learning (MobileNetV2 base)
2. Proper input size matching (critical lesson!)
3. Global Average Pooling
4. Dropout regularization
5. Early Stopping

## Tools Used
- TensorFlow/Keras
- MobileNetV2 (Transfer Learning)
- CIFAR-10 dataset

## Author
Kaviya V | github.com/kaviyavijayan11
