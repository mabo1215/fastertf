import logging
import pickle
logging.getLogger("tensorflow").setLevel(logging.DEBUG)
from tensorflow import keras
import numpy as np
import time
from test_encryptionSupport import Encryption


def ReLU(x):
    return x * (np.invert(x < 0))

## Implementation of proposed search based eReLU activation
#regular RelU is used on top to cover the values that are not
# in the set
def eReLU(x, set):
    return ReLU(x * (np.invert(np.isin(x, set))))

sf = 2**10
enc = Encryption(sf)

## Load shared encrypted relu set
with open('ReLUSet.pkl', 'rb') as f:
    relu_set = pickle.load(f)
set = relu_set[0]

#### Load saved trained model weights ###
with open('mnist_weights9931.pkl', 'rb') as f:
    weights = pickle.load(f)

### Scale used to convert floating point numbers to integer
scale = 2**7

## Convert floating points weights and biases to integers weights
w1 = (weights[0][0]*scale).astype(int)
w2 = (weights[1][0]*scale).astype(int)

wfc1 = (weights[2][0]*scale).astype(int)


b1 = (weights[0][1]*scale).astype(int)
b2 = (weights[1][1]*scale).astype(int)

bfc1 = (weights[2][1]*scale).astype(int)
##########################################
model_conv1 = keras.Sequential([
    keras.layers.InputLayer(input_shape=(28, 28)),
    keras.layers.Reshape(target_shape=(28, 28, 1)),
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), use_bias=True, padding='SAME', name='conv1'),
])
model_conv2 = keras.Sequential([
  keras.layers.InputLayer(input_shape=(28, 28, 32)),
keras.layers.AveragePooling2D(pool_size=(2, 2), padding='SAME', name="pool1"),
  keras.layers.Conv2D(filters=64, kernel_size=(3, 3), use_bias=True, padding='SAME', name='conv2'),
])
model_d1 = keras.Sequential([keras.layers.InputLayer(input_shape=(14, 14, 64)),
    keras.layers.AveragePooling2D(pool_size=(2, 2), padding='SAME', name="pool2"),
  keras.layers.Flatten(name='flat'),
  keras.layers.Dense(10, use_bias=True, name='dense1')])


## the original testing set of MNIST data is avalable in Data adv[0]
## Data_adv[2] contains the corresponding original labels

with open('DNNattack/Data_adv0.pkl', 'rb') as f:
    Data_adv = pickle.load(f)
test_images = Data_adv[0].reshape(10000, 28, 28)
test_labels = np.argmax(Data_adv[2], axis=1)

####################################################################################
##### Generating Timing results for a single data sample form MNIST test dataset
####################################################################################
test_image = test_images[0].reshape(1, 28, 28)
test_label = test_labels[0]



model_conv1.get_layer('conv1').set_weights([w1.astype(float), b1.astype(float)])
model_conv2.get_layer('conv2').set_weights([w2.astype(float), b2.astype(float)])
model_d1.get_layer('dense1').set_weights([wfc1.astype(float), bfc1.astype(float)])



start_time = time.time()
conv1 = model_conv1.predict(test_image)
conv1r = ReLU(conv1)
conv2 = model_conv2.predict(conv1r)
conv2r = ReLU(conv2)
Y = model_d1(conv2r)
time1 = time.time() - start_time

print("Integer model execution time for single data sample --- %s seconds ---" % (time.time() - start_time+time1))
####################################################################################
#####
####################################################################################

####################################################################################
##### Generating Accuracy  and timing results for the MNIST test dataset
####################################################################################

start_time = time.time()
conv1 = model_conv1.predict(test_images)
conv1r = ReLU(conv1)
conv2 = model_conv2.predict(conv1r)
conv2r = ReLU(conv2)
Y = model_d1(conv2r)
time1 = time.time() - start_time

print("Integer model execution time --- %s seconds ---" % (time.time() - start_time+time1))

num_data = len(test_labels)
cp = 0
## Find the index of the highest value of output and compare with label to get final classification accuracy
for i in range(num_data):
    if np.argmax(Y[i])==test_labels[i]:
      cp = cp+1
print("Integer Model Accuracy ")
print(cp/num_data)
print("done")

####################################################################################
#####
####################################################################################