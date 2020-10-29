import logging
import pickle
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

from tensorflow import keras
import numpy as np
import time



#Define the model architecture same as trained model
model = keras.Sequential([
  keras.layers.InputLayer(input_shape=(28, 28)),
  keras.layers.Reshape(target_shape=(28, 28, 1)),
  keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', use_bias=True, padding='SAME', name='conv1'),
  keras.layers.AveragePooling2D(pool_size=(2, 2), padding='SAME', name="pool1"),
  keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', use_bias=True, padding='SAME', name='conv2'),
  keras.layers.AveragePooling2D(pool_size=(2, 2), padding='SAME', name="pool2"),
  keras.layers.Flatten(name='flat'),
  keras.layers.Dense(10, use_bias=True, name='dense1')
])



#### Load saved trained model weights ###
with open('mnist_weights9931.pkl', 'rb') as f:
    weights = pickle.load(f)

############ Add original weights to the neural network
for layer in model.layers:

    if layer.name == 'conv1':

        layer.set_weights([weights[0][0], weights[0][1]])
    if layer.name == 'conv2':

        layer.set_weights([weights[1][0], weights[1][1]])

    if layer.name == 'dense1':

        layer.set_weights([weights[2][0], weights[2][1]])

model.summary()
acc=[]
## Get Original model accuracy for different epsilon 0.01-0.06
for i in range(6):
    with open('DNNattack/Data_adv'+str(i)+'.pkl', 'rb') as f:
        Data_adv = pickle.load(f)
    test_images = Data_adv[1].reshape(10000,28,28)
    test_labels = np.argmax(Data_adv[2], axis=1)


    start_time = time.time()
    Yint = model.predict(test_images)
    cp = 0
    #Find the index of the highest value of output and compare with label to get final classification accuracy
    for i in range(test_images.shape[0]):
        if np.argmax(Yint[i])==test_labels[i]:
          cp = cp+1
    print("Original model execution time --- %s seconds ---" % (time.time() - start_time))
    print(cp/test_images.shape[0])
    acc.append(cp/test_images.shape[0])


print(acc)
