# Image Classifier - Transfer Learning

## Project Overview
Multi-class image classifier using Transfer Learning with 
MobileNetV2, classifying images into 10 CIFAR-10 categories.

## Approach
- Used pretrained MobileNetV2 (trained on ImageNet, 1.4M images)
- Froze base model weights, added custom classification head
- Only trained 165,258 parameters instead of 
  millions - much faster than training from scratch

## Results
- Test Accuracy: 29.80%
- Training completed in 13 epochs
- Trainable params: 6.8% of total

## Key Techniques
1. Transfer Learning (MobileNetV2 base)
2. Global Average Pooling
3. Dropout regularization
4. Early Stopping

## Comparison to From-Scratch CNN (Day 17)
Day 17 MNIST CNN: Built from scratch, 98.91% accuracy, 
simpler grayscale digits, more epochs needed.

This project: Complex color images (10 real-world categories),
faster training via transfer learning, demonstrates 
industry-standard approach for limited data/time scenarios.

## Tools Used
- TensorFlow/Keras
- MobileNetV2 (Transfer Learning)
- CIFAR-10 dataset

## Author
Kaviya V | github.com/kaviyavijayan11
