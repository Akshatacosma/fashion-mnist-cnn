# FashionMNIST Image Classification using CNN

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify clothing images from the FashionMNIST dataset.  
The model is trained and evaluated over multiple epochs with epoch-wise accuracy tracking and visual prediction analysis.

---

## Dataset
- FashionMNIST
- 60,000 training images
- 10,000 test images
- Image size: 28 × 28 (grayscale)
- 10 clothing categories

### Classes
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

---

## Model Architecture
- Two convolutional layers with ReLU activation
- MaxPooling layers for spatial downsampling
- Fully connected layers for classification
- Output layer with 10 classes

---

## Training Details
- Framework: PyTorch
- Optimizer: Adam
- Loss Function: CrossEntropyLoss
- Batch size: 64
- Epochs: 5
- Device: CPU

---

## Evaluation
- Accuracy evaluated after each epoch
- 100 random test images used per epoch
- Visual display of predictions with class names
- Final accuracy achieved around 85–90%

---

## How to Run
```bash
pip install -r requirements.txt
python fashion.py
