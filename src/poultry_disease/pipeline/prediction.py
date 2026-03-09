import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image


class PredictionPipeline:

    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = [
            "Coccidiosis",
            "Healthy",
            "New Castle Disease",
            "Salmonella"
        ]

    def predict(self, image_path):

        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        predictions = self.model.predict(img_array)
        predicted_class = self.class_names[np.argmax(predictions)]

        return predicted_class