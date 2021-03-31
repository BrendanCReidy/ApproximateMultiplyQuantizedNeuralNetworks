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

print(train_images[0][:100])

# Define the model architecture
model = tf.keras.Sequential([
  tf.keras.layers.InputLayer(input_shape=(28, 28)),
  tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
  tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
  tf.keras.layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu'),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(32),
  tf.keras.layers.Dense(10)
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])
# train full precision model
model.fit(
  train_images,
  train_labels,
  epochs=1,
  validation_data=(test_images, test_labels)
)

model.summary()


import ApproximateNeuralNetwork

approxNetwork = ApproximateNeuralNetwork.ApproximateNeuralNetwork(model)

approxNetwork.summary()

num_test = 10 # number of test images to validate on
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

