#!pip install streamlit
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np

# ----------- Styling -----------
st.set_page_config(page_title="Which Bollywood Star Are You?", page_icon="🎬", layout="centered")
st.markdown("""
    <style>
    .stApp {
        background-color: #fef9f4;
        color: #2c2c2c;
    }

    h1, h2, h3 {
        color: #d63384;
        font-family: 'Segoe UI', sans-serif;
    }

    div.stButton > button:first-child {
        background-color: #d6336c;
        color: white;
        font-size: 16px;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        transition: 0.3s;
    }

    div.stButton > button:hover {
        background-color: #a61e4d;
        color: #fff;
        transform: scale(1.02);
    }

    </style>
""", unsafe_allow_html=True)

# ----------- Model Load -----------
detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
feature_list = pickle.load(open('embedding.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# ----------- Helpers -----------
def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('uploads', uploaded_image.name), 'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False

def extract_features(img_path, model, detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)
    x, y, width, height = results[0]['box']
    face = img[y:y + height, x:x + width]
    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = np.asarray(image).astype('float32')
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result

def recommend(feature_list, features):
    similarity = [cosine_similarity(features.reshape(1, -1), f.reshape(1, -1))[0][0] for f in feature_list]
    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

# ----------- Main UI -----------
st.title('🎬 Which Bollywood Celebrity Are You?')

uploaded_image = st.file_uploader('📸 Choose an image')

if uploaded_image is not None:
    if save_uploaded_image(uploaded_image):
        display_image = Image.open(uploaded_image)
        features = extract_features(os.path.join('uploads', uploaded_image.name), model, detector)
        index_pos = recommend(feature_list, features)
        predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))

        col1, col2 = st.columns(2)

        with col1:
            st.subheader('🖼️ Your Uploaded Image')
            st.image(display_image)

        with col2:
            st.subheader(f"🌟 You Resemble: {predicted_actor}")
            st.image(filenames[index_pos], width=300)
