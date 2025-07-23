# Store Shelf Product Detector

Detect products on retail store shelves in images using advanced computer vision models. This project leverages a custom-trained model (improved version of YOLO) for object detection, USEARCH for vector database, and ResNet/CLIP/DinoV2 for image similarity and retrieval, providing an interactive Streamlit web interface for users.

## Features

- **Product Detection:** Upload shelf images and automatically detect products using a trained YOLO model.
- **Visual Results:** Bounding boxes are drawn on products, colored by confidence (green, yellow, red).
- **Streamlit UI:** Simple and interactive web interface for image upload, visualization, and results.
- **Dataset Reference:** Based on SKU110K dataset ([Source](https://datasetninja.com/sku110k)).
