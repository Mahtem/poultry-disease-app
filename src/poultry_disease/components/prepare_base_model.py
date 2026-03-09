import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from poultry_disease.entity.config_entity import PrepareBaseModelConfig
from pathlib import Path


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config


    def get_base_model(self):
        """
        Load VGG16 base model
        """

        self.model = VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        Path(self.config.base_model_path).parent.mkdir(parents=True, exist_ok=True)

        self.model.save(self.config.base_model_path)

        print(f"Base model saved at: {self.config.base_model_path}")


    def update_base_model(self):
        """
        Add custom classification layers
        """

        base_model = tf.keras.models.load_model(self.config.base_model_path)

        base_model.trainable = False

        x = base_model.output
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation="relu")(x)
        predictions = layers.Dense(self.config.params_classes, activation="softmax")(x)

        full_model = models.Model(inputs=base_model.input, outputs=predictions)

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.params_learning_rate
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        Path(self.config.updated_base_model_path).parent.mkdir(parents=True, exist_ok=True)

        full_model.save(self.config.updated_base_model_path)

        print(f"Updated model saved at: {self.config.updated_base_model_path}")