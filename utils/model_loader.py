import os
import torch
import streamlit as st
from ultralytics import YOLO
import easyocr

@st.cache_resource
def load_yolo_models():
    """
    Loads all five traffic enforcement YOLOv8 models from the models/ directory.
    Stores them in a dictionary. Returns None for models that are missing.
    """
    models = {}
    model_paths = {
        'helmet': 'models/helmet.pt',
        'plate': 'models/plate.pt',
        'triple': 'models/triple.pt',
        'seatbelt': 'models/seatbelt.pt',
        'bengaluru': 'models/bengaluru.pt'
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for name, path in model_paths.items():
        if os.path.exists(path):
            try:
                # Load YOLO model
                model = YOLO(path)
                model.to(device)
                
                # Apply Helmets model override:
                # HELMET MODEL NOTE: After loading helmet.pt, override class names:
                # model.names = {0:'numberPlate', 1:'faceWithNoHelmet', 2:'faceWithGoodHelmet', 3:'faceWithBadHelmet', 4:'rider'}
                if name == 'helmet':
                    # Modify in-place as model.names is a read-only dictionary attribute in some ultralytics versions
                    model.names[0] = 'numberPlate'
                    model.names[1] = 'faceWithNoHelmet'
                    model.names[2] = 'faceWithGoodHelmet'
                    model.names[3] = 'faceWithBadHelmet'
                    model.names[4] = 'rider'
                
                models[name] = model
            except Exception as e:
                # Catch any runtime errors loading the model (e.g. corrupted file)
                st.sidebar.error(f"Error initializing model '{name}' from path '{path}': {e}")
                models[name] = None
        else:
            models[name] = None
            
    return models

@st.cache_resource
def load_ocr_reader():
    """
    Loads EasyOCR Reader for English characters. Cache resources so it is loaded once.
    """
    try:
        # Determine device
        use_gpu = torch.cuda.is_available()
        
        # Specify model storage directory within the project
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, 'easyocr_models')
        
        # Initialize EasyOCR
        reader = easyocr.Reader(['en'], gpu=use_gpu, model_storage_directory=model_dir)
        return reader
    except Exception as e:
        st.sidebar.warning(f"EasyOCR could not be initialized: {e}. OCR features will fallback to simulation.")
        return None
