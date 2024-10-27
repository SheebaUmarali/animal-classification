import os
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import array_to_img

# Define CIFAR-10 class names
class_names = [
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"
]

def save_images_by_class(X, y, data_dir="data"):
    """
    Saves each CIFAR-10 image to its respective class folder.
    """
    for i, (image, label) in enumerate(zip(X, y)):
        class_name = class_names[label[0]]
        class_dir = os.path.join(data_dir, class_name)

        # Ensure the class directory exists
        os.makedirs(class_dir, exist_ok=True)

        # Convert array to image and save as PNG
        img = array_to_img(image)
        img_path = os.path.join(class_dir, f"{class_name}_{i}.png")
        img.save(img_path)
        
        # Optionally print progress for every 1000 images saved
        if i % 1000 == 0:
            print(f"Saved {i} images")

if __name__ == "__main__":
    # Load CIFAR-10 data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Combine train and test sets
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    # Save images to their respective class folders
    save_images_by_class(X, y, data_dir="../data")
    print("All images saved by class.")
