import tensorflow as tf

# Path to your model folder (NOT the .pb file directly)
saved_model_dir = r"D:\Poultry_Disease\artifacts\training\model"

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

# Save
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Conversion complete!")