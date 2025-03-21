
import cv2
import lmdb
import pickle
import os
from tqdm import tqdm

def capture_faces(num_faces=1000, face_size=(128, 128)):
    
    name = str(input("Please enter your first name: "))
    output_path = os.path.join("data", f"{name}.lmdb")
    # Initialize face detector
    facedetect = cv2.CascadeClassifier('models\haarcascade_frontalface_default.xml')
    
    # Create LMDB environment
    map_size = num_faces * 1024 * 1024 * 3  
    env = lmdb.open(output_path, map_size=map_size)

    # Initialize webcam
    cam = cv2.VideoCapture(0)
    
    saved_count = 0
    progress = tqdm(total=num_faces, desc="Saving faces")

    try:
        with env.begin(write=True) as txn:
            while saved_count < num_faces:
                ret, frame = cam.read()
                if not ret:
                    continue

                # Detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = facedetect.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    # Extract and resize face
                    face_img = frame[y:y+h, x:x+w]
                    resized_face = cv2.resize(face_img, face_size)

                    # Store in LMDB
                    key = f"face_{saved_count:08d}".encode()
                    txn.put(key, pickle.dumps(resized_face))
                    
                    saved_count += 1
                    progress.update(1)
                    
                    # Display current face count on the frame
                    cv2.putText(frame, str(saved_count), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 1)
        
                    # Draw rectangle around detected face
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
              
                # Show preview
                cv2.imshow('Face Capture', frame)
                
                # Exit on key ('q') pressed or completion (100 faces collected)
                k=cv2.waitKey(1)
                if k==ord('q') or len(faces_data)==1000:
                    break

    finally:
        cam.release()
        cv2.destroyAllWindows()
        progress.close()
        print(f"\nSaved {saved_count} faces")



if __name__ == "__main__":
    # Capture 1000 faces to database
    capture_faces()
   
