# streamlit_app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
import time

# ------------------------
# Load TFLite model
# ------------------------
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ------------------------
# Class names
# ------------------------
class_names = [
    "Coccidiosis",
    "Newcastle Disease",
    "Salmonella",
    "Healthy"
]

# ------------------------
# Streamlit Page Config
# ------------------------
st.set_page_config(
    page_title="🐔 Poultry Disease Predictor Using VGG16",
    page_icon="🐓",
    layout="centered"
)

# ------------------------
# Sidebar
# ------------------------
st.sidebar.header("Instructions")
st.sidebar.info(
    """
    1. Upload a clear image of your chicken or affected area.  
    2. Wait a few seconds while the model analyzes it.  
    3. View the top 3 predicted diseases with probabilities.  
    """
)

# ------------------------
# App Header
# ------------------------
st.markdown(
    "<h1 style='text-align: center; color: darkgreen;'>🐔 Poultry Disease Predictor</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center; color: gray;'>Powered by TensorFlow Lite</h4>",
    unsafe_allow_html=True
)

st.write("---")

# ------------------------
# Image Upload
# ------------------------
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # ------------------------
    # Preprocess Image
    # ------------------------
    input_shape = input_details[0]['shape']
    img_resized = image.resize((input_shape[2], input_shape[1]))
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # ------------------------
    # Inference with Spinner
    # ------------------------
    with st.spinner("Analyzing image..."):
        time.sleep(1)  # Optional delay for UX
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

    # ------------------------
    # Display Top 3 Predictions
    # ------------------------
    top_indices = np.argsort(output)[-3:][::-1]
    top_predictions = {class_names[i]: output[i]*100 for i in top_indices}

    st.subheader("Top Predictions")
    
    # Display as cards
    for disease, prob in top_predictions.items():
        st.markdown(
            f"""
            <div style='
                background-color: #f0f8ff;
                padding: 10px;
                margin: 5px 0;
                border-radius: 8px;
                border: 1px solid #d0d0d0;
            '>
                <h4 style='margin: 0'>{disease}</h4>
                <p style='margin: 0'>Probability: {prob:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Optional: show as DataFrame
    df = pd.DataFrame({
        "Disease": list(top_predictions.keys()),
        "Probability": [f"{p:.2f}%" for p in top_predictions.values()]
    })
    st.write("Or in a table format:")
    st.table(df)
