import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def evaluate():

    # Load trained model
    model = tf.keras.models.load_model(
        "artifacts/training/model.keras"
    )

    # Load test CSV
    test_df = pd.read_csv(
        "artifacts/data_ingestion/test.csv"
    )

    # Image generator
    datagen = ImageDataGenerator(rescale=1./255)

    test_generator = datagen.flow_from_dataframe(
        dataframe=test_df,
        directory="artifacts/data_ingestion/Train",
        x_col="images",
        y_col="label",
        target_size=(224, 224),  # must match training size
        batch_size=32,
        class_mode="categorical",
        shuffle=False
    )

    # Evaluate accuracy
    loss, accuracy = model.evaluate(test_generator)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

    # Predictions
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=list(test_generator.class_indices.keys())
    ))


if __name__ == "__main__":
    evaluate()