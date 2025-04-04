import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the model with caching for efficiency
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('best_model.keras')

# Preprocess function for face detection and resizing
def preprocess(img, face_size=(128, 128)):
    img_array = []
    
    # Initialize face detector
    facedetect = cv2.CascadeClassifier('models\haarcascade_frontalface_default.xml')
    
    # Detect faces
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in face:
        # Extract and resize face
        face_img = img[y:y+h, x:x+w]
        resized_face = cv2.resize(face_img, face_size)
        img_array.append(np.array(resized_face))
    return img_array
    
# Prediction function
def predict(face_dim):
    model = load_model()
    prediction = model.predict(face_dim).astype(int)
    if prediction == 0:
        st.success("Access granted. Welcome KC!", icon="✅")
    elif prediction == 1:
        st.success("Access granted. Welcome Peace!", icon="✅")
    else:
        st.success("Access denied!!!", icon="❌")

# Streamlit UI
st.title("Secret Vault Access")

#key to save face captured
if 'face' not in st.session_state.keys():
    st.session_state['face'] = None
    
# Camera input widget
captured_image = st.camera_input("Scan", label_visibility = "hidden")

if captured_image:
    
    image = Image.open(captured_image)
    #st.image(image)
    st.session_state['face'] = np.array(image.convert('RGB'))

    faces = preprocess(st.session_state['face'])

    if faces:
        # Prepare input for the model
        face_dim = np.expand_dims(faces[0], axis=0)
        # Make prediction using the loaded model
        try:
            predict(face_dim)
        except Exception as e:
            st.error("Failed! Please try again", icon="🚨")
    else:
        st.warning("No faces detected!", icon="🚨")
