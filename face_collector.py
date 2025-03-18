import cv2
import numpy as np
import os
 
name = str(input("Please enter your first name: "))

video=cv2.VideoCapture(0)

facedetect=cv2.CascadeClassifier('models\haarcascade_frontalface_default.xml')

faces_data = []  # Empty list to store resized face images
i = 0  # Counter to track processed frames

while True:
    # Read a frame from the video source (webcam/file)
    ret,frame=video.read()
    
    # Convert frame to grayscale (face detection works better on grayscale)
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the haar cascade pre-trained classifier
    faces=facedetect.detectMultiScale(gray, 1.3 ,5)
    
    for (x, y, w, h) in faces:
        # Crop the face region from the frame
        crop_img = frame[y:y+h, x:x+w, :]
        
        # Resize cropped face to 50x50 pixels 
        resized_img = cv2.resize(crop_img, (50, 50))
        
        # Collect every 10th face until 100 faces are stored
        if len(faces_data) <= 100 and i % 100 == 0:
            faces_data.append(resized_img)
            
        # Increment frame counter
        i += 1
        
        # Display current face count on the frame
        cv2.putText(frame, str(len(faces_data)), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 1)
        
        # Draw rectangle around detected face
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)

    # Show the processed frame
    cv2.imshow("Frame", frame)
    
    # Check for exit key ('q') or completion (100 faces collected)
    k=cv2.waitKey(1)
    if k==ord('q') or len(faces_data)==10:
        break

video.release()
cv2.destroyAllWindows()

#faces_data=np.asarray(faces_data)

import psycopg2 

# Connect to the postgreSQL database 
def create_connection(): 
    conn = psycopg2.connect(dbname='FaceRecog', 
                            user='postgres', 
                            password=password,
                            host='localhost', 
                            port='5432') 
    # Get the cursor object from the connection object 
    curr = conn.cursor() 
    return conn, curr 
  
def create_table(): 
    
    try: 
        # Get the cursor object from the connection object 
        conn, curr = create_connection() 
        try: 
            # Fire the CREATE query 
            curr.execute("DROP TABLE authorized") 
            curr.execute("CREATE TABLE IF NOT EXISTS authorized(personID TEXT, name TEXT, faceImg BYTEA)") 
              
        except(Exception, psycopg2.Error) as error: 
            # Print exception 
            print("Error while creating authorized table", error) 
        finally: 
            # Close the connection object 
            conn.commit() 
            conn.close() 
    finally: 
        # Since we do not have to do anything here we will pass 
        pass
  
def write_blob(personID,face,name): 
    try:  
        # Read database configuration 
        conn, cursor = create_connection() 
        try:            
            # Execute the INSERT statement 
            # Convert the image data to Binary 
            cursor.execute("INSERT INTO authorized (personID,name,faceImg) " + "VALUES(%s,%s,%s)", (personID,name, psycopg2.Binary(face))) 
            # Commit the changes to the database 
            conn.commit() 
        except (Exception, psycopg2.DatabaseError) as error: 
            print("Error while inserting data in authorized table", error) 
        finally: 
            # Close the connection object 
            conn.close() 
    finally: 
        # Since we do not have to do anything here we will pass 
        pass
print("Saved faces")
        
# Call the create table method       
create_table() 
# Prepare sample data, of images, from local drive 
for i, face in enumerate(faces_data):
    personID = name+str(i)
    write_blob(personID,face,name)

import pandas as pd
def read_img(): 
    conn, curr = create_connection() 
    curr.execute("SELECT * FROM authorized")
    rows = curr.fetchall()
    df = pd.DataFrame(rows, columns=["PersonID","name","face"])
    return df
    
df = read_img()
print(df)

import matplotlib.pyplot as plt
import numpy
import base64
head = df.head()
def Display_images(head):
    plt.figure(figsize=(10, 10))
    for images in head.face:
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            blob = base64.b64encode(images[i]).decode('utf-8')
            #plt.title(class_names[labels[i]])
            plt.axis("off")
            
Display_images(df)
