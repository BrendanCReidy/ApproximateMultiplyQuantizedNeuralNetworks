import numpy as np
import math

import tensorflow as tf
from tensorflow import keras

from keras import Model
from keras.layers import Dense, Dropout, Input

def binarize(data):
    a = data > 0.5
    b = a.astype(int)
    return b

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = binarize(train_images)
test_images = binarize(test_images)

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

padded_train_images = np.zeros((60000,32,32))
padded_test_images = np.zeros((10000,32,32))
for i in range(train_images.shape[0]):
    padded_train_images[i] = np.pad(train_images[i], 2, pad_with)

for i in range(test_images.shape[0]):
    padded_test_images[i] = np.pad(test_images[i], 2, pad_with)

print(padded_train_images.shape)
print(padded_test_images.shape)

print(padded_train_images[0][:100])

train_images = padded_train_images
test_images = padded_test_images

# Define the model architecture
model = tf.keras.Sequential([
  tf.keras.layers.InputLayer(input_shape=(32, 32)),
  tf.keras.layers.Reshape(target_shape=(32, 32, 1)),
  tf.keras.layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu'),
  tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2)),
  tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
  tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(120),
  tf.keras.layers.Dense(84),
  tf.keras.layers.Dense(10)
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])
# train full precision model
"""
model.fit(
  train_images,
  train_labels,
  epochs=5,
  validation_data=(test_images, test_labels)
)
#"""

model = keras.models.load_model("LeNetTrained.h5")

#model.save("LeNetTrained.h5")

num_test = 1000 # number of test images to validate on

model.summary()

y_pred = model.predict(test_images[:num_test])
y_pred = np.argmax(y_pred, axis=1) 
acc = tf.keras.metrics.Accuracy()
acc.reset_states()
acc.update_state(y_pred, test_labels[:num_test])
print("Old acc", acc.result().numpy())


import ApproximateNeuralNetwork

approxNetwork = ApproximateNeuralNetwork.ApproximateNeuralNetwork(model, nofbits=16, mult_wd=4)

approxNetwork.summary()

predictions = np.zeros((num_test,10))

for i in range(num_test):
  print("Test input", i)
  y_pred = approxNetwork.predict(test_images[i])
  predictions[i] = y_pred

print("Num bits:", approxNetwork.nofbits)
print("Acc mult bits:",approxNetwork.mult_wd)

y_pred = np.argmax(predictions, axis=1) 
acc = tf.keras.metrics.Accuracy()
acc.reset_states()
acc.update_state(y_pred, test_labels[:num_test])
print("New acc", acc.result().numpy())

y_pred = model.predict(test_images[:num_test])
y_pred = np.argmax(y_pred, axis=1) 
acc = tf.keras.metrics.Accuracy()
acc.reset_states()
acc.update_state(y_pred, test_labels[:num_test])
print("Old acc", acc.result().numpy())

