import streamlit as st
import cv2
import numpy as np
import re
from ultralytics import YOLO
from doctr.models import ocr_predictor
# from PIL import Image

# --- CONFIGURATION & MODELS ---
st.set_page_config(page_title="ANPR Dashboard", layout="wide")

@st.cache_resource
def load_models():
    # Initialize DocTR (PyTorch backend)
    ocr_model = ocr_predictor(pretrained=True)
    # Load your custom YOLO model
    yolo_model = YOLO('runs/detect/train/weights/best.pt') 
    return yolo_model, ocr_model

yolo_model, ocr_model = load_models()

def clean_plate_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

# --- UI LAYOUT ---
st.title("🚗 Intelligent Number Plate Recognition")
st.markdown("Upload a vehicle image to detect and extract the license plate number.")

uploaded_file = st.sidebar.file_uploader("Choose a vehicle image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    display_img = img.copy()
    
    with st.spinner('Detecting and Reading Plate...'):
        results = yolo_model(img)
        detections = []

        for result in results:
            for box in result.boxes:
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # Preprocessing for OCR: Crop with 10% padding
                h, w, _ = img.shape
                pad = int((x2 - x1) * 0.1)
                plate_crop = img[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]
                
                # Convert BGR to RGB for DocTR
                plate_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                
                # OCR Inference
                out = ocr_model([plate_rgb])
                export = out.export()
                
                plate_contents = []
                for page in export['pages']:
                    for block in page['blocks']:
                        for line in block['lines']:
                            for word in line['words']:
                                clean_txt = clean_plate_text(word['value'])
                                if len(clean_txt) >= 3: # Adjusted for short plates
                                    plate_contents.append(clean_txt)
                
                final_plate_text = " ".join(plate_contents)
                
                # Draw Bounding Box & Text on image
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(display_img, final_plate_text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                
                detections.append({
                    "text": final_plate_text,
                    "confidence": conf,
                    "crop": plate_rgb
                })

    # --- DISPLAY RESULTS ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Detection Result")
        st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col2:
        st.subheader("Extracted Details")
        if detections:
            for i, det in enumerate(detections):
                st.write(f"**Plate {i+1}:** `{det['text']}`")
                st.write(f"Confidence: {det['confidence']:.2f}")
                st.image(det['crop'], caption=f"Cropped Plate {i+1}", width=150)
                st.divider()
        else:
            st.warning("No license plates detected.")

else:
    st.info("Please upload an image from the sidebar to begin.")