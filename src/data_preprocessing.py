import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def get_data_generator():
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2]
    )
    return datagen


def load_data(data_directory, image_size=(32, 32)):
    """
    Load images and labels from the specified directory.
    Assumes that images are organized in subdirectories named by class.
    """
    images = []
    labels = []
    class_names = os.listdir(data_directory)

    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_directory, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=image_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img_array)
            labels.append(label)

    return np.array(images), np.array(labels), class_names

def preprocess_data(data_directory):
    # Load the data
    X, y, class_names = load_data(data_directory)

    # Normalize pixel values to be between 0 and 1
    X = X.astype('float32') / 255.0

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    return (X_train, y_train), (X_val, y_val), class_names

if __name__ == "__main__":
    # Example usage
    data_directory = 'C:\\animal-classification\\data'  # Specify your dataset path here
    (X_train, y_train), (X_val, y_val), class_names = preprocess_data(data_directory)
    print("Data preprocessing complete.")
