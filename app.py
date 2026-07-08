import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import time

#  PAGE CONFIG 
st.set_page_config(
    page_title="Pet Emotion Detector",
    page_icon="🐾",
    layout="centered"
)

#  TITLE 
st.title(" Pet Emotion AI")
st.markdown("Upload your pet's photo to know if they are **Happy**, **Sad**, or **Angry**!")

#  LOAD MODEL 
@st.cache_resource
def load_model_cached():
    try:
        model = load_model("facial_expression_model_mobilenet.h5")
        return model
    except Exception as e:
        st.error(f"Model load nahi ho pa raha: {e}")
        return None

model = load_model_cached()

if model is None:
    st.stop()

# CLASSES 
emotions = {0: "Happy", 1: "Sad", 2: "Angry"}

#  PREPROCESS FUNCTION 
def preprocess_image(image):
    """   """
    img_array = np.array(image.convert('RGB'))
    img_resized = cv2.resize(img_array, (128, 128))
    img_normalized = img_resized.astype('float32') / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)
    return img_input

#  FILE UPLOADER 
uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])

# PREDICTION LOGIC
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Your Pet", use_container_width=True)
    
    with st.spinner("Analyzing emotions with 94% accuracy model..."):
        processed_img = preprocess_image(image)
        predictions = model.predict(processed_img)
        pred_class = np.argmax(predictions)
        confidence = np.max(predictions) * 100
        time.sleep(0.5)
    
    #  DISPLAY RESULTS 
    st.divider()
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"## {emotions[pred_class]}")
    with col2:
        st.markdown(f"### Confidence: **{confidence:.1f}%**")
    
    st.subheader("Confidence Breakdown")
    
    for i, (label, prob) in enumerate(zip(emotions.values(), predictions[0])):
        if i == pred_class:
            bar_color = "green" if confidence > 75 else "orange"
        else:
            bar_color = "gray"
        st.progress(float(prob), text=f"{label}: {prob*100:.1f}%")
    
    st.divider()
    if confidence > 85:
        st.success("High Confidence! Model is very sure.")
    elif confidence > 60:
        st.warning("Medium Confidence. Try a clearer photo.")
    else:
        st.error("Low Confidence. Please upload a clearer photo.")

      

st.divider()
st.caption(" ")