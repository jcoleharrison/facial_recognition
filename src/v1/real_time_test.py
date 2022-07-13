# Import standard dependencies
import cv2
import os
import time
import random
import numpy as np
# from matplotlib import pyplot as plt
# Import tensorflow dependencies - Functional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

IMG_SIZE = 900


def preprocess(file_path):

    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image
    img = tf.io.decode_jpeg(byte_img)

    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    # Scale image to be between 0 and 1
    img = img / 255.0

    # Return image
    return img


def verify(model, verification_threshold, detection_threshold):

    # Build results array
    results = []
    for image in os.listdir('application_data/verification_images'):
        input_img = preprocess(os.path.join(
            'application_data', 'input_image', 'input_image.jpg'))
        validation_img = preprocess(os.path.join(
            'application_data', 'verification_images', image))

        # Make predictions
        result = model.predict(
            list(np.expand_dims([input_img, validation_img], axis=1)))
        results.append(result)

    detection = np.sum(np.array(results))  # > detection_threshold)
    verification = detection / \
        len(os.listdir(os.path.join('application_data', 'verification_images')))
    verified = verification  # > verification_threshold

    return results, verified

    # Metric above which a prediction is considered a match
    # Verification Threshold: Proporation of positive predictions / total predictions


class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


# def main():
# Reload model
siamese_model = tf.keras.models.load_model('src/siamesemodelv3.h5',
                                           custom_objects={'L1Dist': L1Dist, 'BinaryCrossentropy': tf.losses.BinaryCrossentropy})

# Load in the last frame
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[100:100+900, 300:300+900, :]

    cv2.imshow('Verification', frame)

    # Verification trigger
    if cv2.waitKey(10) & 0xFF == ord('v'):
        # Save input image to application_data/input_image folder
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # h, s, v = cv2.split(hsv)

        # lim = 255 - 10B
        # v[v > lim] = 255
        # v[v <= lim] -= 10

        # final_hsv = cv2.merge((h, s, v))
        # img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        cv2.imwrite('application_data/input_image/input_image.jpg', frame)
        # Run verification
        results, verified = verify(siamese_model, 0.7, 0.7)
        print(verified)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# main()
