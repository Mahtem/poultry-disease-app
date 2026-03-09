from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

def train(self):

    # Build base model directly
    base_model = VGG16(
        input_shape=self.config.params_image_size,
        weights="imagenet",
        include_top=False
    )

    base_model.trainable = False

    # Add custom layers
    x = base_model.output
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    predictions = layers.Dense(
        self.config.params_classes,
        activation="softmax"
    )(x)

    model = models.Model(
        inputs=base_model.input,
        outputs=predictions
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=self.config.params_learning_rate
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )