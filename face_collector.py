import streamlit as st
import cv2
import numpy as np
import os
from supabase import create_client, Client
from streamlit_webrtc import webrtc_streamer
import av


# Initialize Supabase client from Streamlit secrets
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)


st.title("Face Capture from Video App")

# Get user name
name = st.text_input("Enter your first name:")

# Initialize frame counter and faces data list
i = 0
faces_data = []

def capture(frame: av.VideoFrame):
    video = frame.to_ndarray(format="bgr24")

    facedetect = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
        # Run the video capture loop as long as the user is in the application
    
    #while True:
        
         # Convert frame to grayscale (face detection works better on grayscale)
         #gray=cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
        
         # Detect faces using the haar cascade pre-trained classifier
         #faces=facedetect.detectMultiScale(gray, 1.3 ,5)
        
         
            #cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)

         

         #if cv2.waitKey(1) & 0xFF == ord('q') or len(faces_data) >= 100:
          #  break

         
     return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="streamer",
    video_frame_callback=capture,
    sendback_audio=False
    )
# Upload faces data to Supabase Storage
j = 1
for face in faces_data:
               try:
                   response = supabase.storage.from_("faces").upload(
                       file=face,
                       path=f"{name}/{name}{j}.png",
                       file_options={"content-type": "image/png"}
                   )
                   
                   j += 1
               except Exception as e:
                   st.error(f"Error uploading image {j}: {e}")

               st.success(f"Successfully uploaded {len(faces_data)} faces!")



