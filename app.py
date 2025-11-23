from ultralytics import YOLO
import streamlit as st
import torch
import torchvision
from PIL import Image
from pathlib import Path


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
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")
        detect_object(uploaded_file)


def detect_object(uploaded_file):
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    results = model.predict(uploaded_file.name, conf=0.5, save=True, save_txt=True)
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
    
    
