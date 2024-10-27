import tensorflow as tf
from tensorflow.keras import layers, models
from data_preprocessing import preprocess_data

def build_model(input_shape, num_classes):
    """
    Build and compile the CNN model with Batch Normalization and Dropout.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')  # Number of classes
    ])

    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def train_model(X_train, y_train, X_val, y_val, class_names):
    """
    Train the model using the training and validation datasets.
    """
    model = build_model((32, 32, 3), len(class_names))  # Adjust input shape if necessary

    # Set up data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    model.fit(datagen.flow(X_train, y_train, batch_size=32),
              validation_data=(X_val, y_val),
              epochs=50,  # Adjust epochs as necessary
              verbose=1)

    # Save the model
    model.save('models/best_model.h5')
    print("Model training complete and saved to models/best_model.h5.")

if __name__ == "__main__":
    data_directory ='C:\\animal-classification\\data'   # Specify your dataset path here
    (X_train, y_train), (X_val, y_val), class_names = preprocess_data(data_directory)
    train_model(X_train, y_train, X_val, y_val, class_names)
