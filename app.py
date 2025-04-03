import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

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
        st.sucess("Access granted. Welcome KC!")
    elif prediction == 1:
        st.sucess("Access granted. Welcome Peace!")
    else:
        st.sucess("Access denied!!!")

# Streamlit UI
st.title("Secret Vault Access")

# Button to initiate photo capture
if st.button("Scan"):
    # Camera input widget
    captured_image = st.camera_input("Capture your face")

    if captured_image is not None:
        # Convert captured image to OpenCV format
        bytes_data = captured_image.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # Preprocess the image for face detection and resizing
        faces = preprocess(img)

        if faces:
            # Prepare input for the model
            face_dim = np.expand_dims(faces[0], axis=0)
            # Make prediction using the loaded model
            try:
                prediction = predict(face_dim)
            except Exception as e:
                st.error(f"Failed! Please try again")
        else:
            st.warning("No faces detected!")
