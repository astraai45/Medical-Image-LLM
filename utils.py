import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical
import cv2
import os

def create_keras_model():
    model = Sequential([
        Input(shape=(64, 64, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(6, activation='softmax')
    ])
    return model

# Save model weights
def save_model_weights(model, file_path='model.weights.h5'):
    model.save_weights(file_path)
    print(f"Model weights saved to {file_path}")

# Load model weights
def load_model_weights(model, file_path='model.weights.h5'):
    if os.path.exists(file_path):
        model.load_weights(file_path)
        print(f"Model weights loaded from {file_path}")
        return True
    else:
        raise FileNotFoundError(f"Model weights file {file_path} not found")
    return False

# Preprocess a single image
def preprocess_image(image_path, img_size=(64, 64)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(image, img_size)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=-1)
    return image

# Load new data
def load_new_data(new_data_path, img_size=(64, 64), classes=6, label_mapping=None):
    files_path = []
    files_labels = []
    for class_folder in os.listdir(new_data_path):
        class_path = os.path.join(new_data_path, class_folder)
        if os.path.isdir(class_path):
            for file in os.listdir(class_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    files_path.append(os.path.join(class_path, file))
                    files_labels.append(class_folder)
    
    preprocessed_images = []
    labels = []
    for file_path, label in zip(files_path, files_labels):
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        image = cv2.resize(image, img_size)
        image = image.astype('float32') / 255.0
        preprocessed_images.append(image)
        labels.append(label_mapping[label])
    
    images = np.array(preprocessed_images)
    labels = np.array(labels)
    labels_onehot = to_categorical(labels, num_classes=classes)
    return images, labels_onehot

# Load and preprocess dataset (for test subset)
def load_and_preprocess_data(data_path, img_size=(64, 64), classes=6, limit_samples=None):
    files_path = []
    files_labels = []
    label_mapping = {'AbdomenCT': 0, 'BreastMRI': 1, 'Hand': 2, 'CXR': 3, 'HeadCT': 4, 'ChestCT': 5}
    
    for class_folder in os.listdir(data_path):
        class_path = os.path.join(data_path, class_folder)
        if os.path.isdir(class_path):
            for file in os.listdir(class_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    files_path.append(os.path.join(class_path, file))
                    files_labels.append(class_folder)
    
    preprocessed_images = []
    labels = []
    sample_count = 0
    
    for file_path, label in zip(files_path, files_labels):
        if limit_samples and sample_count >= limit_samples:
            break
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        image = cv2.resize(image, img_size)
        image = image.astype('float32') / 255.0
        preprocessed_images.append(image)
        labels.append(label_mapping[label])
        sample_count += 1
    
    images = np.array(preprocessed_images)
    labels = np.array(labels)
    labels_onehot = to_categorical(labels, num_classes=classes)
    
    shuffle_indices = np.random.permutation(len(images))
    shuffled_images = images[shuffle_indices]
    shuffled_labels = labels_onehot[shuffle_indices]
    
    return shuffled_images, shuffled_labels, label_mapping