import logging
import pickle
logging.getLogger("tensorflow").setLevel(logging.DEBUG)
from tensorflow import keras
import numpy as np
import time
from test_encryptionSupport import Encryption


def ReLU(x):
    return x * (np.invert(x < 0))

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

acc=[]
## Get Encrypted model accuracy for different epsilon 0.01-0.06
for i in range(6):
    with open('DNNattack/Data_adv'+str(i)+'.pkl', 'rb') as f:
        Data_adv = pickle.load(f)
    test_images = Data_adv[1].reshape(10000,28,28)
    test_labels = np.argmax(Data_adv[2], axis=1)


    start_time = time.time()
    ew11 = np.empty_like(w1)
    ew12 = np.empty_like(w1)
    eb11 = np.empty_like(b1)
    eb12 = np.empty_like(b1)
    for i in range(32):
        [ew1, ew2] = enc.encrpy_matrix_to_pair(w1[:, :, 0, i].reshape((3, 3)), 1)
        ew11[:, :, 0, i] = ew1
        ew12[:, :, 0, i] = ew2

        [eb1, eb2] = enc.encrpy_matrix_to_pair(b1[i].reshape((1, 1)), 1)
        eb11[i] = eb1
        eb12[i] = eb2

    print("encryption--- %s seconds ---" % (time.time() - start_time))

    ### Create first encrypted neural network

    model_conv1.get_layer('conv1').set_weights([ew11.astype(float), eb11.astype(float)])
    model_conv2.get_layer('conv2').set_weights([w2.astype(float), b2.astype(float)])
    model_d1.get_layer('dense1').set_weights([wfc1.astype(float), bfc1.astype(float)])

    # Get output Y1 to test_images from first encrypted neural network

    eset1 = relu_set[1]
    test_image = test_images[0].reshape((1,28,28))
    start_time = time.time()
    conv1 = model_conv1.predict(test_images)
    conv1r = eReLU(conv1, eset1)
    conv2 = model_conv2.predict(conv1r)
    conv2r = eReLU(conv2, eset1)
    Y1 = model_d1(conv2r)
    time1 = time.time() - start_time
    ### Create 2nd encrypted neural network

    # Get output Y1 to test_images from 2nd encrypted neural network

    model_conv1.get_layer('conv1').set_weights([ew12.astype(float), eb12.astype(float)])
    model_conv2.get_layer('conv2').set_weights([w2.astype(float), b2.astype(float)])
    model_d1.get_layer('dense1').set_weights([wfc1.astype(float), bfc1.astype(float)])

    eset2 = relu_set[2]
    start_time = time.time()
    conv1 = model_conv1.predict(test_images)
    conv1r = eReLU(conv1, eset2)
    conv2 = model_conv2.predict(conv1r)
    conv2r = eReLU(conv2, eset2)
    Y2 = model_d1(conv2r)

    print("encrypted model execution time --- %s seconds ---" % (time.time() - start_time+time1))
    Y = np.zeros(Y1.shape)

    start_time = time.time()
    ## Decrypting of output values
    for i in range(10000):

        for k in range(10):
            ## Combine each element of the output from two neural networks and Decrypt to get the plaintext output layer values
            eypair = [[Y1[i][k]], [Y2[i][k]]]
            Y[i, k] = enc.decrpt_pair(eypair, 1)[0]

    print("decryption time --- %s seconds ---" % (time.time() - start_time))
    num_data = len(test_labels)
    cp = 0
    ## Find the index of the highest value of output and compare with label to get final classification accuracy
    for i in range(num_data):
        if np.argmax(Y[i])==test_labels[i]:
          cp = cp+1
    print("Encrypted Accuracy ")
    print(cp/num_data)
    print("done")
    acc.append(cp/num_data)
print(acc)
