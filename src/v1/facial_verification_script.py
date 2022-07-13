# Import Dependencies

import pickle
import cv2
import os
import time
import random
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.metrics import Precision, Recall

# Import tensorflow dependencies - Functional API
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten, Dropout, Concatenate
import tensorflow as tf
from tensorflow.keras import backend as K

os.chdir('src')

# Create Folder Structures

POS_PATH = os.path.join('..', 'data', 'positive')
NEG_PATH = os.path.join('..', 'data', 'negative')
ANC_PATH = os.path.join('..', 'data', 'anchor')

os.makedirs(POS_PATH, exist_ok=True)
os.makedirs(NEG_PATH, exist_ok=True)
os.makedirs(ANC_PATH, exist_ok=True)

TAKE_N = 300
IMG_SIZE = 900
EPOCHS = 10

# 3. Load and Preprocess Images


anchor = tf.data.Dataset.list_files(ANC_PATH+'/*.jpg').take(TAKE_N)
positive = tf.data.Dataset.list_files(POS_PATH+'/*.jpg').take(TAKE_N)
negative = tf.data.Dataset.list_files(NEG_PATH+'/*.jpg').take(TAKE_N)


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


positives = tf.data.Dataset.zip(
    (anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip(
    (anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)


def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)


# Build dataloader pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)


# Training partition
train_data = data.take(round(len(data)*.95))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)


# Testing partition
test_data = data.skip(round(len(data)*.95))
test_data = test_data.take(round(len(data)*.05))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)


# 4. Model Engineering

def make_embedding():
    inp = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='input_image')

    # first block
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D(pool_size=(8, 8), padding='same')(c1)
    dr1 = Dropout(0.3)(m1)
    c1_flat = Flatten()(dr1)

    # second block
    c2 = Conv2D(128, (7, 7), activation='relu')(dr1)
    m2 = MaxPooling2D(pool_size=(6, 6), padding='same')(c2)
    dr2 = Dropout(0.3)(m2)

    # third block
    c3 = Conv2D(128, (4, 4), activation='relu')(dr2)
    m3 = MaxPooling2D(pool_size=(2, 2), padding='same')(c3)
    dr3 = Dropout(0.3)(m3)

    # final embedding block
    c4 = Conv2D(256, (4, 4), activation='relu')(dr3)
    f1 = Flatten()(c4)
    #combined = Concatenate()([f1, c1_flat])
    d1 = Dense(4096, activation='sigmoid',
               kernel_regularizer=tf.keras.regularizers.l2(l=.01))(f1)

    return Model(inputs=[inp], outputs=[d1], name='embedding')


"""Testing with pretrained weights"""


def pretrained_embeddings(shape, embedding=1024, fineTune=False, fine_tune_at=None):
    inputs = tf.keras.layers.Input(shape)
    base_model = tf.keras.applications.vgg16.VGG16(
        input_shape=shape, include_top=False, weights='imagenet')
    base_model.summary()
    if fineTune == False:
        base_model.trainable = False
    else:
        base_model.trainable = True
    # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
    x = base_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(embedding)(x)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model


embedding = pretrained_embeddings(
    (IMG_SIZE, IMG_SIZE, 3), fineTune=True, fine_tune_at=16)
#embedding = make_embedding()
embedding.summary()


# 4.2 Build Distance Layer

class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


# 4.3 Make Siamese Model

def make_siamese_model():

    # Anchor image input in the network
    input_img = Input(name='input_img', shape=(IMG_SIZE, IMG_SIZE, 3))

    # Validation image in the network
    validation_img = Input(name='validation_img',
                           shape=(IMG_SIZE, IMG_SIZE, 3))

    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_img), embedding(validation_img))

    # Classification layer

    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_img, validation_img], outputs=classifier, name='SiameseNetwork')


siamese_model = make_siamese_model()
siamese_model.summary()


# 5. Training

# 5.1 Setup Loss and Optimizer

binary_cross_loss = tf.losses.BinaryCrossentropy()

"""Testing with contrastive loss"""


def contrastive_loss(y, preds, margin=1):
    # explicitly cast the true class label data type to the predicted
    # class label data type
    y = tf.cast(y, preds.dtype)
    # calculate the contrastive loss between the true labels and
    # the predicted labels
    squaredPreds = K.square(preds)
    squaredMargin = K.square(K.maximum(margin - preds, 0))
    loss = 1-K.mean(y * squaredPreds + (1 - y) * squaredMargin)
    return loss


opt = tf.keras.optimizers.Adam(learning_rate=.00006)


# 5.2 Establish Checkpoints

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)


# 5.3 Build Train Step Function

@tf.function
def train_step(batch):

    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        # Get anchor and positive/negative image
        X = batch[:2]
        # Get Label
        y = batch[2]

        # Forward pass
        yhat = siamese_model(X, training=True)

        # Calculate loss
        # loss = binary_cross_loss(y, yhat)
        loss = contrastive_loss(y, yhat)

    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)

    # Calculate update weights and apply to siamese models
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

    return loss


# 5.4 Build Training Loop

# Import metric calculations


def train(data, epochs):
    # Loop over epochs
    losses = []
    for epoch in range(1, epochs + 1):
        print('Epoch {}/{}'.format(epoch, epochs))
        progbar = tf.keras.utils.Progbar(len(data))

        start = time.time()  # record start time

        # Loop over batches
        for idx, batch in enumerate(data):
            loss = train_step(batch)
            losses.append(loss)
            progbar.update(idx+1)

        end = time.time()  # record end time

        print('Epoch {} Loss {:.4f} Time {:.4f}'.format(epoch, loss, end - start))
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

    return losses


# 5.5 Train the model

losses = train(train_data, EPOCHS)

siamese_model.save('siamesemodelv3.h5')
# 6. Evaluate Model


# 6.2 Make Predictions

# Get a batch of test data
batches = test_data.as_numpy_iterator()


test_input, test_val, y_true = batches.next()


# Make predictions
yhat = siamese_model.predict([test_input, test_val])
yhat


# Post processing the results
[1 if prediction > 0.5 else 0 for prediction in yhat]


# ## 6.3 Calculate Metrics

# Create a metric object
r = Recall()
p = Precision()

# Calculating recall value
r.update_state(y_true, yhat)
p.update_state(y_true, yhat)

# Return recall value
print('recall: ', r.result().numpy())
print('precision: ', p.result().numpy())


# ## 6.4 Save and Viz Results

with open('loss', 'wb') as file:
    pickle.dump(losses, file)

plt.plot(losses)
