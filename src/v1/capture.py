import tensorflow as tf
import uuid
import os
import numpy as np
import cv2



POS_PATH = os.path.join('..', 'data', 'positive')
NEG_PATH = os.path.join('..', 'data', 'negative')
ANC_PATH = os.path.join('..', 'data', 'anchor')


def capture():
    cap = cv2.VideoCapture(0)
    last_frame = 0
    while cap.isOpened(): 
        ret, frame = cap.read()
    
        # Cut down frame to 250x250px
        frame = frame[100:100+900,300:300+900, :]
        
        # Collect anchors 
        if cv2.waitKey(1) & 0XFF == ord('a'):
            # Create the unique file path 
            imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
            # Write out anchor image
            cv2.imwrite(imgname, frame)
        
        # Collect positives
        if cv2.waitKey(1) & 0XFF == ord('p'):
            # Create the unique file path 
            imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
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
    return last_frame

_last = capture()

np.save('../test_images/last_frame.npy', _last)