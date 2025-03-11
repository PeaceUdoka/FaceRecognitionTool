import streamlit as st
import cv2
import numpy as np
from supabase import create_client, Client
from PIL import Image

# Load secrets from Streamlit secrets management
supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_KEY"]
supabase = create_client(supabase_url, supabase_key)

st.title("Face Capture App")

# Get user name
name = st.text_input("Enter your first name:")

# Initialize variables
faces_data = []
i = 0

# Webcam capture button
if st.button("Start Face Capture") and name:
    
    facedetect = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    
    stframe = st.empty()
    status_text = st.empty()

    while True:
        video = st.camera_input()
        
        if not video:
            st.error("Failed to capture")
            break

        gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50))
            
            if len(faces_data) <= 100 and i % 10 == 0:
                faces_data.append(resized_img)
            
            i += 1
            
            cv2.putText(frame, str(len(faces_data)), (50,50), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 1)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)

        # Display frame in Streamlit
        stframe.image(frame, channels="BGR")
        
        # Check completion conditions
        if len(faces_data) >= 100:
            video.release()
            cv2.destroyAllWindows()
            break

    # Upload to Supabase after collection
    if faces_data:
        j = 1
        for face in faces_data:
            response = supabase.storage.from_("faces").upload(
                file=face,
                path=f"{name}/{name}{j}.png",
                file_options={"content-type": "image/png"}
            )
            j += 1
        
        st.success(f"Successfully uploaded {len(faces_data)} faces!")
