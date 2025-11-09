import os
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

# --- Configuration ---
# Determine the base directory of this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model path updated to look for 'my_model.h5' directly in the 'backend' folder
MODEL_PATH = os.path.join(BASE_DIR, "my_model.h5") 
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Parameters matching the model's training configuration
TARGET_SIZE = (256, 256)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

app = FastAPI(title="Potato Disease Detector API")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Configure CORS (FastAPI standard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Loading ---

MODEL = None
try:
    # Load the Keras model using the absolute path
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# --- Utility Function ---

def read_file_as_image(data: bytes) -> np.ndarray:
    """
    Converts raw image bytes from an uploaded file into a preprocessed numpy array.
    """
    try:
        image = Image.open(io.BytesIO(data))
        image = image.resize(TARGET_SIZE)
        image_array = np.array(image)

        if image_array.ndim == 2:
            image_array = np.stack((image_array,) * 3, axis=-1)
        elif image_array.shape[-1] == 4:
            image_array = image_array[..., :3]
        
        # NOTE: If your model requires normalization (e.g., dividing by 255),
        # uncomment the line below.
        # image_array = image_array / 255.0

        return image_array
    except Exception as e:
        print(f"Error processing image: {e}")
        raise ValueError("Invalid image file or format.")

# --- Routes ---

def get_server_status() -> str:
    """Helper to determine model status."""
    return "Model Loaded Successfully" if MODEL else "Model Loading Failed (Check path/file)"

@app.get("/", response_class=HTMLResponse)
async def index_get(request: Request):
    """Route to serve the index.html template (default view)."""
    # Renders the page with only the status, no prediction result yet
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "status": get_server_status(), "result": None}
    )

@app.post("/", response_class=HTMLResponse)
async def index_post(request: Request, file: UploadFile = File(...)):
    """Route to handle image upload from the form, make prediction, and display result."""
    
    result: Dict[str, Any] = {}
    
    if not MODEL:
        result = {"error": "Prediction service is unavailable. Model failed to load."}
    
    elif not file or file.filename == '':
        result = {"error": "No file selected for upload."}
    
    else:
        try:
            # Read image data
            # file.read() is asynchronous in FastAPI
            image_bytes = await file.read() 
            image = read_file_as_image(image_bytes)

            # Add batch dimension and make prediction
            img_batch = np.expand_dims(image, 0)
            prediction = MODEL.predict(img_batch, verbose=0) 
            
            # Extract results
            predicted_class_index = np.argmax(prediction[0])
            predicted_class = CLASS_NAMES[predicted_class_index]
            confidence = float(np.max(prediction[0]))

            result = {
                "class": predicted_class,
                "confidence": round(confidence * 100, 2)
            }

        except ValueError as ve:
            result = {"error": str(ve)}
        except Exception as e:
            print(f"An unexpected error occurred during prediction: {e}")
            result = {"error": "An internal error occurred during prediction."}
            
    # Always render the template, passing the result (or error)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "status": get_server_status(), "result": result}
    )

if __name__ == "__main__":
    # Note: We run FastAPI with uvicorn directly
    print("FastAPI server starting...")
    import uvicorn
    # Use the --reload flag for development mode
    uvicorn.run(app, host="127.0.0.1", port=8001)

