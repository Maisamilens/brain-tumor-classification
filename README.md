Brain Tumor Classification in MRI Images Using CNN and Transfer Learning

This repository contains the code and notebooks for the research project:

“Brain Tumor Classification in MRI Images Using Combined Transfer Learning and Convolutional Neural Networks”
Authors: Maisam Abbas, Muhammad Hassan, Ran-Zan Wang*, Chin-Hung Teng
Corresponding Author: Ran-Zan Wang – rzwang@saturn.yzu.edu.tw

🧠 Project Overview

Brain tumor detection and classification is critical for early diagnosis and effective treatment. This project implements a deep learning framework for multi-class MRI brain tumor classification. It combines:

A custom-designed Convolutional Neural Network (CNN)
Six pre-trained transfer learning models (InceptionV3, EfficientNetV2L, ResNet152V2, Xception, VGG16, MobileNetV2)
Ensemble strategies to improve prediction accuracy

Our approach achieves state-of-the-art results on the Kaggle-Multiclass brain MRI dataset, with 99.54% accuracy for the custom CNN and over 99% precision, recall, and F1-score for top models.

📦 Repository Contents
File	Description
README.md	Project description and instructions
custom_cnn.ipynb.ipynb	Notebook for building and training custom CNN
pretrained_ensemble.ipynb.ipynb	Notebook for evaluating pre-trained models and ensembles
requirements.txt	Python dependencies
📊 Dataset

The dataset is a combination of Figshare, Br35H, and cleaned sources. It contains 7023 MRI images categorized into 4 classes:

Glioma Tumor – Cancerous tumor in glial cells
Meningioma Tumor – Non-cancerous tumor originating from meninges
No Tumor – Normal brain scans
Pituitary Tumor – Tumors affecting the pituitary gland

The dataset link: Brain Tumor MRI Dataset on Kaggle

Note: The "No Tumor" images were obtained from the Br35H dataset.

⚙️ Setup & Installation
Install Dependencies
pip install --upgrade tensorflow
pip install --upgrade tensorflow[and-cuda]
pip install -r requirements.txt
Import Required Packages
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix
🔄 Data Processing
Load and Label Dataset
def get_data_labels(directory, shuffle=True, random_state=0):
    from sklearn.utils import shuffle
    data_path, data_index = [], []
    label_dict = {label: index for index, label in enumerate(sorted(os.listdir(directory)))}
    for label, index in label_dict.items():
        label_dir = os.path.join(directory, label)
        for img in os.listdir(label_dir):
            data_path.append(os.path.join(label_dir, img))
            data_index.append(index)
    if shuffle:
        data_path, data_index = shuffle(data_path, data_index, random_state=random_state)
    return data_path, data_index
Create TensorFlow Dataset
def parse_function(filename, label, image_size, n_channels):
    img_string = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img_string, n_channels)
    img = tf.image.resize(img, image_size)
    return img, label

def get_dataset(paths, labels, image_size, n_channels=1, batch_size=32):
    path_ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = path_ds.map(lambda p, l: parse_function(p, l, image_size, n_channels), tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
🏋️ Training Setup
Image size: 168 × 168
Channels: Grayscale (1 channel)
Classes: 4
Batch size: 32
Random seed: 111
GPU enabled using TensorFlow
train_paths, train_labels = get_data_labels("/kaggle/input/brain-tumor-mri-dataset/Training")
test_paths, test_labels = get_data_labels("/kaggle/input/brain-tumor-mri-dataset/Testing")

train_ds = get_dataset(train_paths, train_labels, (168,168))
test_ds = get_dataset(test_paths, test_labels, (168,168))
📈 Results
Model	Accuracy	Precision	Recall	F1-score
Custom CNN	99.54%	99.55%	99.52%	99.53%
EfficientNetV2L	99.47%	99.48%	99.46%	99.47%
InceptionV3	99.39%	99.41%	99.37%	99.39%
Ensemble 1	99.47%	99.48%	99.46%	99.47%
🧩 Notebooks
custom_cnn.ipynb.ipynb – Build and train custom CNN model
pretrained_ensemble.ipynb.ipynb – Evaluate pre-trained models & ensemble strategies
📫 Contact
Maisam Abbas – s1129105@mail.yzu.edu.tw
🔗 References
Abbas, M., Hassan, M., Wang, R.-Z., Teng, C.-H. Brain Tumor Classification in MRI Images Using Combined Transfer Learning and Convolutional Neural Networks. Submitted to Journal of Imaging, 2026.
