import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# --- SAYFA AYARI ---
st.set_page_config(page_title="Capsule Detector", layout="centered")

# --- MODELİ 1 KEZ YÜKLE (performans için) ---
@st.cache_resource
def load_model():
    return YOLO("D:/AI_Project/dataset/runs/detect/train-3/weights/best.pt")

model = load_model()

# --- LETTERBOX (ZOOM YOK) ---
def letterbox_no_zoom(im, new_shape=(640, 640), color=(0, 0, 0)):
    h, w = im.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    r = min(r, 1.0)

    new_w, new_h = int(w * r), int(h * r)
    im_resized = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_w = new_shape[1] - new_w
    pad_h = new_shape[0] - new_h

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    return cv2.copyMakeBorder(
        im_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )

# --- UI ---
st.title("☕ Capsule Defect Detection System (YOLOv8)")
st.write("Upload a coffee capsule image to detect damaged and intact capsules.")

uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded:
    # --- Görseli oku ---
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img0 = cv2.imdecode(file_bytes, 1)

    # --- Letterbox ---
    img = letterbox_no_zoom(img0, (640, 640))

    # --- Tahmin ---
    results = model(img, conf=0.5)

    # --- Çiz ---
    out = results[0].plot()
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    # --- Göster ---
    st.image(out, caption="Detection Result", width=700)

    # --- INFO ---
    st.success("Detection completed successfully ✅")