import os
import io
import zipfile
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "my_model.h5")
TEMPLATES_ZIP_PATH = os.path.join(BASE_DIR, "templates.zip")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

TARGET_SIZE = (256, 256)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# --- Unzip templates if needed ---
if not os.path.exists(TEMPLATES_DIR) and os.path.exists(TEMPLATES_ZIP_PATH):
    with zipfile.ZipFile(TEMPLATES_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(TEMPLATES_DIR)
    st.success("✅ Templates extracted successfully!")

# --- Load model ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

MODEL = load_model()

# --- Utility function ---
def read_file_as_image(data: bytes) -> np.ndarray:
    """Converts uploaded file bytes into a preprocessed numpy array."""
    image = Image.open(io.BytesIO(data))
    image = image.resize(TARGET_SIZE)
    image_array = np.array(image)

    # Convert grayscale to RGB
    if image_array.ndim == 2:
        image_array = np.stack((image_array,) * 3, axis=-1)
    elif image_array.shape[-1] == 4:
        image_array = image_array[..., :3]

    return image_array

# --- Streamlit UI ---
st.set_page_config(page_title="Potato Leaf Disease Detector", layout="wide")
st.title("🥔 Potato Leaf Disease Detector")
st.write("Upload an image of a potato leaf, and the model will predict the disease.")

# Sidebar uploader and preview
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display uploaded image in sidebar
        st.sidebar.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        # Preprocess image and predict
        image_array = read_file_as_image(uploaded_file.read())
        img_batch = np.expand_dims(image_array, 0)
        prediction = MODEL.predict(img_batch, verbose=0)
        predicted_class_index = np.argmax(prediction[0])
        predicted_class = CLASS_NAMES[predicted_class_index]
        confidence = float(np.max(prediction[0]))

        # Show results in main page
        st.success(f"Predicted Class: **{predicted_class}**")
        st.info(f"Confidence: **{round(confidence*100, 2)}%**")

    except Exception as e:
        st.error(f"Error processing the image: {e}")

# Optional: Show model status if no file is uploaded
else:
    st.info("Upload an image to get prediction results.")
