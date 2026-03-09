import tensorflow as tf

# Load existing .keras model
model = tf.keras.models.load_model("artifacts/training/model.keras")

# Save in TensorFlow SavedModel format (folder, no extension)
model.save("artifacts/training/model")

print("Model successfully re-saved in SavedModel format.")