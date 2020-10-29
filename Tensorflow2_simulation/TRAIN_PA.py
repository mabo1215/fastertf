import logging
import pickle
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf
from tensorflow import keras


# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# use integer data 0-10
train_images = ((train_images/255*10).astype(int)).astype(float)
test_images = ((test_images/255*10).astype(int)).astype(float)

# Define the model architecture
model = keras.Sequential([
  keras.layers.InputLayer(input_shape=(28, 28)),
  keras.layers.Reshape(target_shape=(28, 28, 1)),
  keras.layers.Conv2D(filters=32, kernel_size=(3, 3),  use_bias=True, padding='SAME'),
  #keras.layers.Activation(tf.nn.relu),
  keras.layers.Activation(tf.math.square),
  keras.layers.AveragePooling2D(pool_size=(2, 2), padding='SAME'),
  keras.layers.Dropout(0.4),
  keras.layers.Conv2D(filters=64, kernel_size=(3, 3),  use_bias=True, padding='SAME'),
  #keras.layers.Activation(tf.nn.relu),
  keras.layers.Activation(tf.math.square),
  keras.layers.AveragePooling2D(pool_size=(2, 2), padding='SAME'),
  keras.layers.Dropout(0.4),
  keras.layers.Flatten(),
  keras.layers.Dense(10, use_bias=True)
])

# Train the digit classification model
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(
  train_images,
  train_labels,
  epochs=15,
  validation_data=(test_images, test_labels)
)


print(history.history)

#Save Model weights and biases to pickel file
weights = []
for layer in model.layers:

    w = layer.get_weights()
    if len(w)>0:
      print(layer.name)
      weights.append(w)


with open('mnist_weights_PA_20.pkl', 'wb') as f:
  pickle.dump(weights, f)