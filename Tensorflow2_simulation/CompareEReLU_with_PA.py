import numpy as np
import pickle
import time

from test_encryptionSupport import Encryption
sf = 2**8
enc = Encryption(sf)

## Calculating average execution time for cypher domain square function for 10000 samples ##
dts=[]
for p in range(10):
    dt=0
    for i in range(10000):
        w = np.array([i])
       # print(w)
        #w = np.random.randint(-128*128, 128*128, (16,))
        ew = enc.one_way_encrypt_vector(w, 1)
        tic = time.time()
        ew_sq = enc.e_square_pair([[ew[0]], [ew[1]]], 1)
        dt = dt+time.time()-tic

        # Decryption of the squared values
        #eypair = [[ey0[0][k]], [ey1[0][k]]]
        #w1 = enc.esup.s_decrypt(ew_sq).astype('float')
            #w1  = enc.decrpt_pair([ew_sq[0], ew_sq[1]], 1)[0]
            #print(w1)
            #print("done")
    print("encrpyted square calculation time: ", dt)
    dts.append(dt)

print(np.mean(dts))
print(np.std(dts))
##################################################################


##################################################################
## Calculating execution time EReLU for 10000 samples, when the
# Set size increase from 10^2 to 10^8##
######################################################
X = np.random.randint(0,high=10000,size =(10000,), dtype=int)
dts= []
for i in range(2,8,1):
    set = range(0,10**i,1)
    tic = time.time()
    T = np.isin(X, set)
    dt = time.time()-tic
    print(dt)
    dts.append(dt)