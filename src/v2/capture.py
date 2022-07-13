import tensorflow as tf
import uuid
import os
import numpy as np
import cv2


os.chdir('src/v2/')
INTAKE_PATH = os.path.join('..','..', 'data', 'intake')
os.makedirs(INTAKE_PATH, exist_ok=True)


def capture():
    cap = cv2.VideoCapture(0)
    last_frame = 0
    while cap.isOpened(): 
        ret, frame = cap.read()
    
        # Cut down frame to 250x250px
        frame = frame[100:100+900,300:300+900, :]
        
        # Collect positives
        if cv2.waitKey(1) & 0XFF == ord('p'):
            # Create the unique file path 
            imgname = os.path.join(INTAKE_PATH, '{}.jpg'.format(uuid.uuid1()))
            # Write out positive image
            cv2.imwrite(imgname, frame)
        
        # Show image back to screen
        cv2.imshow('Image Collection', frame)
        
        # Breaking gracefully
        if cv2.waitKey(1) & 0XFF == ord('q'):
            last_frame = frame
            break
            

    # Release the webcam
    cap.release()
    # Close the image show frame
    cv2.destroyAllWindows()


capture()