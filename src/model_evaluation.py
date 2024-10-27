
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from data_preprocessing import preprocess_data

def evaluate_model(model, X_val, y_val, class_names):
    """
    Evaluate the model and display the classification report and confusion matrix.
    """
    y_pred = np.argmax(model.predict(X_val), axis=-1)

    print("Classification Report:")
    print(classification_report(y_val, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png')
    plt.show()
    print("Confusion matrix saved as results/confusion_matrix.png.")

if __name__ == "__main__":
    data_directory = 'C:\\animal-classification\\data'  # Specify your dataset path here
    (X_train, y_train), (X_val, y_val), class_names = preprocess_data(data_directory)
    model = load_model('models/best_model.h5')
    evaluate_model(model, X_val, y_val, class_names)