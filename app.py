# app.py
import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
import os
import clip

from usearch.index import Index
from ultralytics import YOLO
import torchvision.transforms as transforms
import torchvision.models as models
import pickle

# --- Configuration ---
# This dictionary holds all the information for our different models.
# It makes the code clean and easy to add new models in the future.
SIMILARITY_MODELS = {
    "CLIP (ViT-L/14)": {
        "index_path": "shelf_images_clip_l14.usearch",
        "map_path": "image_map_clip_l14.pkl",
        "type": "clip",
        "model_name": "ViT-L/14"
    },
    "DinoV2 (ViT-L/14)": {
        "index_path": "shelf_images_dino_L14.usearch",
        "map_path": "image_map_dino_L14.pkl",
        "type": "dino",
        "model_name": "dinov2_vitl14"
    }
}
DETECTION_MODEL_PATH = 'best.pt'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model Loading (with Streamlit Caching) ---
# @st.cache_resource is a decorator that tells Streamlit to run this function only once.
# This prevents reloading the slow models on every user interaction.

@st.cache_resource
def load_yolo_model():
    print("Loading YOLO detection model...")
    model = YOLO(DETECTION_MODEL_PATH)
    return model

@st.cache_resource
def load_clip_model(model_name):
    print(f"Loading CLIP model: {model_name}...")
    model, preprocess = clip.load(model_name, device=DEVICE)
    return model, preprocess

@st.cache_resource
def load_dino_model(model_name):
    print(f"Loading DinoV2 model: {model_name}...")
    model = torch.hub.load('facebookresearch/dinov2', model_name, verbose=False)
    model.to(DEVICE).eval()
    return model

@st.cache_resource
def load_search_index(index_path, map_path):
    if not os.path.exists(index_path) or not os.path.exists(map_path):
        return None, None
    print(f"Loading search index: {index_path}...")
    index = Index.restore(index_path)
    with open(map_path, 'rb') as f:
        image_map = pickle.load(f)
    return index, image_map

# --- Helper & Drawing Functions ---

def get_color_by_confidence(confidence):
    """Returns a BGR color tuple based on confidence score."""
    if confidence >= 0.80: return (0, 255, 0)      # Green
    elif confidence >= 0.60: return (0, 255, 255)  # Yellow
    else: return (0, 0, 255)                      # Red

def draw_colored_boxes(result):
    """Manually draws bounding boxes with custom colors on an image."""
    image = result.orig_img.copy()
    for box_data in result.boxes:
        box, conf, cls = box_data.xyxy[0], box_data.conf[0].item(), int(box_data.cls[0].item())
        x1, y1, x2, y2 = map(int, box)
        color = get_color_by_confidence(conf)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        label = f"{result.names[cls]} {conf:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - text_height - baseline - 2), (x1 + text_width, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return image

def get_embedding_for_image(pil_image, model_type, model_info):
    """Generic function to get an embedding for an uploaded image."""
    if model_type == 'clip':
        model, preprocess = load_clip_model(model_info['model_name'])
        image_input = preprocess(pil_image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            features = model.encode_image(image_input)
            features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()
        
    elif model_type == 'dino':
        model = load_dino_model(model_info['model_name'])
        transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        image_tensor = transform(pil_image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            return model(image_tensor).cpu().numpy().flatten()
    return None

# --- Main Streamlit App ---

st.set_page_config(layout="wide")
st.title("Store Shelf Product Detector")
st.write("Upload an image of a store shelf to detect products")

# Load the YOLO model once
yolo_model = load_yolo_model()

uploaded_file = st.file_uploader("Upload a shelf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to a PIL Image
    pil_image = Image.open(uploaded_file).convert("RGB")
    
    # Use two columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(pil_image, use_container_width=True)

    with col2:
        st.subheader("Product Detection Results")
        with st.spinner("Detecting products..."):
            # The YOLO model needs a NumPy array
            numpy_image = np.array(pil_image)
            results = yolo_model(numpy_image, verbose=False)
            
            # Draw the colored boxes
            annotated_image = draw_colored_boxes(results[0])
            
            st.image(annotated_image, caption="Detected products with confidence-based colors.", use_container_width=True, channels="BGR")
