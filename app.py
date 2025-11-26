from ultralytics import YOLO
import streamlit as st
import torch
import torchvision
from PIL import Image
from pathlib import Path
import cv2
import tempfile
import numpy as np

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print("using ", device)


model = YOLO("yolov8n.pt")


def main():
    st.markdown("""
        <h1 style='text-align:center; color:white;'>Smart Pantry Food Detector</h1>
        <p style='text-align:center; font-size:18px;'>
            Upload an image of your fridge or pantry and instantly detect the food items inside!
        </p>
        <hr style="border:1px solid #ddd;">
    """, unsafe_allow_html=True)

    st.info("Detected food categories include: **banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake**")
    st.write()
    uploaded_file = st.file_uploader(
        "Choose an image or video",
        type=["png", "jpg", "jpeg", "mp4", "mov", "avi"]
    )
    
   if uploaded_file is not None:
        file_type = "video" if uploaded_file.name.split('.')[-1].lower() in ["mp4", "mov", "avi"] else "image"
        display_uploaded_file(uploaded_file, file_type=file_type)
        if file_type == "image":
            detect_image(uploaded_file)
        elif file_type == "video":
            detect_video(uploaded_file)

def display_uploaded_file(file, file_type):
    if file_type == "image":
        image = Image.open(file)
        st.image(image, caption="Uploaded Image")
    elif file_type == "video":
        st.video(file, caption="Uploaded Video)
    

def detect_image(uploaded_file):
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    results = model.predict(uploaded_file.name, conf=0.5, save=True, save_txt=True)
    st.image(results[0].plot(), caption="Detected Objects")
    
    num_food = len(results[0].boxes.cls)
    st.write(f"Number of food items detected: {num_food}")

    pantry_list(results)

def detect_video(uploaded_file):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video.")
        return

    #choosing 5th frame of video, if not available choosing 1st frame
    target_frame_index = 4  
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= target_frame_index:
        st.warning("5th frame not available, using first frame.")
        target_frame_index = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
    success, frame = cap.read()
    if not success:
        st.error("Error: Could not read frame.")
        cap.release()
        return

    cap.release()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_frame = Image.fromarray(frame_rgb)
    st.image(pil_frame, caption=f"Frame {target_frame_index + 1}")

    results = model.predict(pil_frame, conf=0.5, save=True, save_txt=True)
    st.image(results[0].plot(), caption="Detected Objects")
    
    num_food = len(results[0].boxes.cls)
    st.write(f"Number of food items detected: {num_food}")
    
    pantry_list(results)

def pantry_list(results):
    CATEGORIES = {
        "fruits": ["apple", "banana", "orange"],
        "vegetables": ["carrot", "broccoli"],
        "snacks": ["pizza", "donut", "cake", "hotdog", "sandwich"],
    }
    
    # Get detected classes (as string labels)
    detected_labels = [model.names[int(cls)] for cls in results[0].boxes.cls]
    detected_labels = list(set(detected_labels))
    # Organize into categories
    categorized = {category: [] for category in CATEGORIES}
    for item in detected_labels:
        for category, items in CATEGORIES.items():
            if item in items:
                categorized[category].append(item)
    
    # Display category-wise list
    st.write("### Category-wise Pantry Check")
    for category, items in categorized.items():
        st.write(f"**{category.capitalize()}**: {', '.join(items) if items else 'None detected'}")

if __name__ == "__main__":
    main()
    
