import tensorflow as tf


import numpy as np
import copy
import pickle

path = "./enc2_16_5/"
class EncrptionSuport:
     def __init__(self, aBound, eBound, w, c_ones, M_keys, T_keys, v_onehot, M_onehot):
         self.aBound = aBound
         self.eBound = eBound
         self.w = w
         self.c_ones = c_ones
         self.T_keys = T_keys
         self.v_onehot = v_onehot
         self.M_onehot = M_onehot
         self.M_keys = M_keys

     def keySwitch(self, M, c, l):
         c_star = self.getBitVector(c, l)
         return M.dot(c_star)

     def getRandomMatrix(self, row, col, bound):
         A = np.zeros((row, col))
         for i in range(row):
             for j in range(col):
                 A[i][j] = np.random.randint(bound)
         return A

     def getBitMatrix(self, S, l):
         S_star = list()
         for i in range(l):
             S_star.append(S * 2 ** (l - i - 1))
         S_star = np.array(S_star).transpose(1, 2, 0).reshape(len(S), len(S[0]) * l)
         return S_star

     def getSecretKey(self, T):
         assert (T.ndim == 2)
         I = np.eye(len(T))  # num rows
         return self.hCat(I, T)

     def hCat(self, A, B):
         return np.concatenate((A, B), 1)

     def vCat(self, A, B):
         return np.concatenate((A, B), 0)

     def keySwitchMatrix(self, S, T, l):
         S_star = self.getBitMatrix(S, l)
         A = self.getRandomMatrix(T.shape[1], S_star.shape[1], self.aBound)
         E = self.getRandomMatrix(S_star.shape[0], S_star.shape[1], self.eBound)
         return self.vCat(S_star + E - T.dot(A), A)

     def encrypt(self, T, x, w, l):
         return self.keySwitch(self.keySwitchMatrix(np.eye(len(x)), T, l), w * x, l)

     def addVectors(self, c1, c2):
         return c1 + c2

     def linearTransform(self, M, c, l):
         return M.dot(self.getBitVector(c, l)).astype('int64')

     def linearTransformClient(self, G, S, T, l):
         return self.keySwitchMatrix(G.dot(S), T, l)

     def vectorize(self, M):
         ans = np.zeros((len(M) * len(M[0]), 1))
         for i in range(len(M)):
             for j in range(len(M[0])):
                 ans[i * len(M[0]) + j][0] = M[i][j]
         return ans

     def decrypt(self, S, c, w):
         Sc = S.dot(c)
         return (Sc / w).astype('float').round().astype('int')

     def innerProdClient(self, T, l):
         S = self.getSecretKey(T)
         tvsts = self.vectorize(S.T.dot(S)).T
         mvsts = self.copyRows(tvsts, len(T))
         return self.keySwitchMatrix(mvsts, T, l)

     def copyRows(self, row, numrows):
         ans = np.zeros((numrows, len(row[0])))
         for i in range(len(ans)):
             for j in range(len(ans[0])):
                 ans[i][j] = row[0][j]

         return ans

     def innerProd(self, c1, c2, M, l):
         cc1 = np.zeros((len(c1), 1))
         for i in range(len(c1)):
             cc1[i][0] = c1[i]

         cc2 = np.zeros((1, len(c2)))
         for i in range(len(c2)):
             cc2[0][i] = c2[i]

         cc = self.vectorize(cc1.dot(cc2))

         bv = self.getBitVector((cc / self.w).round().astype('int64'), l)

         return M.dot(bv)

     def one_way_encrypt_vector(self, vector, l, scaling_factor=1):
         padded_vector = np.random.rand(len(vector) + 1)
         padded_vector[0:len(vector)] = vector

         vec_len = len(padded_vector)

         M_temp = (self.M_keys[vec_len - 2].T * padded_vector * scaling_factor / (vec_len - 1)).T
         e_vector = self.innerProd(self.c_ones[vec_len - 2], self.c_ones[vec_len - 2], M_temp, l)
         return e_vector.astype('int')

     def load_linear_transformation(self, syn0_text, T, l, scaling_factor=1):
         syn0_text *= scaling_factor
         return self.linearTransformClient(syn0_text.T, self.getSecretKey(T), T, l)

     def s_decrypt(self, vec):
         return self.decrypt(self.getSecretKey(self.T_keys[len(vec) - 2]), vec, self.w)

     def add_vectors(self, x, y, scaling_factor=10000):
         return x + y

     def transpose(self, syn1, l, scaling_factor):
         rows = len(syn1)
         cols = len(syn1[0]) - 1

         max_rc = max(rows, cols)

         syn1_c = list()
         for i in range(len(syn1)):
             tmp = np.zeros(max_rc + 1)
             tmp[:len(syn1[i])] = syn1[i]
             syn1_c.append(tmp)

         syn1_c_transposed = list()

         for row_i in range(cols):
             syn1t_column = self.innerProd(syn1_c[0], self.v_onehot[max_rc - 1][row_i], self.M_onehot[max_rc - 1][0], l) / scaling_factor
             for col_i in range(rows - 1):
                 syn1t_column += self.innerProd(syn1_c[col_i + 1], self.v_onehot[max_rc - 1][row_i], self.M_onehot[max_rc - 1][col_i + 1], l) / scaling_factor

             syn1_c_transposed.append(syn1t_column[0:rows + 1])

         return syn1_c_transposed

     def int2bin(self, x):
         s = list()
         mod = 2
         while (x > 0):
             s.append(int(x % 2))
             x = int(x / 2)
         return np.array(list(reversed(s))).astype('int64')

     def getBitVector(self, c, l):
         m = len(c)
         c_star = np.zeros(l * m, dtype='int64')
         for i in range(m):
             local_c = int(c[i])
             if (local_c < 0):
                 local_c = -local_c
             b = self.int2bin(local_c)
             if (c[i] < 0):
                 b *= -1
             if (c[i] == 0):
                 b *= 0
             #         try:
             c_star[(i * l) + (l - len(b)): (i + 1) * l] += b
         #         except:
         #             print(len(b))
         #             print(i)
         #             print(len(c_star[(i * l) + (l-len(b)): (i+1) * l]))
         return c_star


################################################
class Encryption:
    def __init__(self, scaling_factor):

        with open(path+'metaData.pkl', 'rb') as f:
            metaData = pickle.load(f)

        with open(path+'c_ones.pkl', 'rb') as f:
            self.c_ones = pickle.load(f)

        with open(path+'T_keys.pkl', 'rb') as f:
            self.T_keys = pickle.load(f)

        with open(path+'M_keys.pkl', 'rb') as f:
            self.M_keys = pickle.load(f)

        with open(path+'M_onehot.pkl', 'rb') as f:
            self.M_onehot = pickle.load(f)

        with open(path+'v_onehot.pkl', 'rb') as f:
            self.v_onehot = pickle.load(f)

        with open(path+'onehot.pkl', 'rb') as f:
            self.onehot = pickle.load(f)

        with open(path+'H_sigmoid_txt.pkl', 'rb') as f:
            self.H_sigmoid_txt = pickle.load(f)

        with open(path+'H_sigmoid.pkl', 'rb') as f:
            self.H_sigmoid = pickle.load(f)

        self.l = metaData[0]
        self.w = metaData[1]
        self.aBound = metaData[2]
        self.tBound = metaData[3]
        self.eBound = metaData[4]
        self.max_dim = metaData[5]
        #self.scaling_factor = metaData[6]
        self.scaling_factor = scaling_factor

        self.esup = EncrptionSuport(self.aBound, self.eBound, self.w, self.c_ones, self.M_keys, self.T_keys, self.v_onehot, self.M_onehot)
############################################
    # def  innerProd(self, c1, c2, M):
    #         return self.esup.innerProd(c1, c2, M, self.l)
    def transpose(self, syn1, scaling_factor):
        return self.esup.transpose(syn1, self.l, scaling_factor)

    def load_linear_transformation(self, syn0_text,scaling_factor = 1000):
        syn0_text *= scaling_factor
        return self.esup.linearTransformClient(syn0_text.T,self.esup.getSecretKey(self.T_keys[len(syn0_text)-1]),self.T_keys[len(syn0_text)-1],self.l)


    def outer_product(self, x, y, scaling_factor):
        flip = False
        if (len(x) < len(y)):
            flip = True
            tmp = x
            x = y
            y = tmp

        y_matrix = list()

        for i in range(len(x) - 1):
            y_matrix.append(y)

        y_matrix_transpose = self.esup.transpose(y_matrix, self.l, scaling_factor)

        outer_result = list()
        for i in range(len(x) - 1):
            outer_result.append(self.mat_mul_forward(x * self.onehot[len(x) - 1][i], y_matrix_transpose, scaling_factor))

        if (flip):
            return self.esup.transpose(outer_result, self.l, scaling_factor)

        return outer_result


    def mat_mul_forward(self, layer_1, syn1, scaling_factor):
        input_dim = len(layer_1)
        output_dim = len(syn1)

        buff = np.zeros(max(output_dim + 1, input_dim + 1))
        buff[0:len(layer_1)] = layer_1
        layer_1_c = buff

        syn1_c = list()
        for i in range(len(syn1)):
            buff = np.zeros(max(output_dim + 1, input_dim + 1))
            buff[0:len(syn1[i])] = syn1[i]
            syn1_c.append(buff)

        layer_2 = self.esup.innerProd(syn1_c[0], layer_1_c, self.M_onehot[len(layer_1_c) - 2][0], self.l) / float(scaling_factor)
        for i in range(len(syn1) - 1):
            layer_2 += self.esup.innerProd(syn1_c[i + 1], layer_1_c, self.M_onehot[len(layer_1_c) - 2][i + 1], self.l) / float(scaling_factor)
        return layer_2[0:output_dim + 1]


    def elementwise_vector_mult(self, x, y, scaling_factor):
        y = [y]

       # one_minus_layer_1 = self.esup.transpose(y, self.l, scaling_factor)

        outer_result = list()
        for i in range(len(x) - 1):
            outer_result.append(self.mat_mul_forward(x * self.onehot[len(x) - 1][i], y, scaling_factor))

        return self.esup.transpose(outer_result, self.l, scaling_factor)[0]


    def get_sigmoid_coefs(self, N):
        coefs = [tf.constant(np.ones(N)*-17 / 80640.0, dtype=tf.float32), tf.constant(np.ones(N)*0.0, dtype=tf.float32), tf.constant(np.ones(N)*1/480.0, dtype=tf.float32), tf.constant(np.ones(N)*0.0, dtype=tf.float32),
                 tf.constant(np.ones(N)*-1 / 48.0, dtype=tf.float32), tf.constant(np.ones(N)*0.0, dtype=tf.float32), tf.constant(np.ones(N)*1 / 4.0, dtype=tf.float32), tf.constant(np.ones(N)*0.5, dtype=tf.float32)]
        return coefs

    def sigmoid_tf(self, x, coefs):
      y = tf.math.polyval(coeffs=coefs, x=x)
      #y = 0.5 + x,1/4.0 - np.power(x, 3) / 48.0 + np.power(x, 5) / 480.0 - np.power(x, 7) / 80640.0 * 17.0
      return y

    def sigmoid(self, x):
      y = 0.5 + x / 4.0 - np.power(x, 3) / 48.0 #+ np.power(x, 5) / 480.0 - np.power(x, 7) / 80640.0 * 17.0
      # print(np.isnan(y))
      # y[np.isnan(y)] = 0
      return y
    def e_square_pair(self, ev_pair, input_scaling_factor):
        ### VERY IMPORTANT #########################
        ## When Decrpyting the output with "decrpt_pair()",
        ## set scaling_factor = (input_scaling_factor^2)
        ############################################
        e_sq0 = np.empty_like(ev_pair[0])
        e_sq1 = np.empty_like(ev_pair[0])
        en1 = self.one_way_encrypt_vector([1], 1).astype('int64')
        e_h = np.stack(ev_pair, axis=-1)
        if e_h.ndim == 2:
            for i, e_pair in enumerate(e_h):
                M_position = self.M_onehot[len(e_pair) - 2][0]
                e_pair2 = self.esup.innerProd(e_pair, e_pair, M_position, self.l)
                e_sq0[i] = e_pair2[0]
                e_sq1[i] = e_pair2[1]

        return [e_sq0, e_sq1]

    def e_sigmoid_pair(self, ev_pair, input_scaling_factor):
        ### VERY IMPORTANT #########################
        ## When Decrpyting the output with "decrpt_pair()",
        ## set scaling_factor = (input_scaling_factor^3 * 10^2)
        ############################################
        e_sig0 = np.empty_like(ev_pair[0])
        e_sig1 = np.empty_like(ev_pair[0])
        en1 = self.one_way_encrypt_vector([1], 1).astype('int64')
        e_h = np.stack(ev_pair, axis=-1)
        if e_h.ndim == 2:
            for i, e_pair in enumerate(e_h):
                M_position = self.M_onehot[len(e_pair) - 2][0]
                e_pair2 = self.esup.innerProd(e_pair, e_pair, M_position, self.l)
                e_pair3 = self.esup.innerProd(e_pair2, e_pair, M_position, self.l)
                e_sig = en1 * 50*(input_scaling_factor**3) + 25*(input_scaling_factor**2)* e_pair - 2 * e_pair3
                e_sig0[i] = e_sig[0]
                e_sig1[i] = e_sig[1]

        return [e_sig0, e_sig1]

    def e_sigmoid(self, layer_2_c):
        out_rows = list()
        for position in range(len(layer_2_c) - 1):
            M_position = self.M_onehot[len(layer_2_c) - 2][0]

            layer_2_index_c = self.esup.innerProd(layer_2_c, self.v_onehot[len(layer_2_c) - 2][position], M_position, self.l) / self.scaling_factor

            x = layer_2_index_c
            x2 = self.esup.innerProd(x, x,  M_position, self.l) / self.scaling_factor
            x3 = self.esup.innerProd(x, x2, M_position, self.l) / self.scaling_factor
            x5 = self.esup.innerProd(x3,x2, M_position,self.l) / self.scaling_factor
            x7 = self.esup.innerProd(x5,x2, M_position,self.l) / self.scaling_factor

            xs = copy.deepcopy(self.v_onehot[3][0])
            xs[1] = x[0]
            # xs[2] = x2[0]
            xs[2] = x3[0]
            #xs[3] = x5[0]
            #xs[4] = x7[0]

            out = self.mat_mul_forward(xs, self.H_sigmoid[0:1], self.scaling_factor)
            out_rows.append(out)
        return self.esup.transpose(out_rows, self.l, self.scaling_factor)[0]

    def one_way_encrypt_vector(self, y, scaling_factor):
        return self.esup.one_way_encrypt_vector(y, self.l, scaling_factor).astype('int64')

    def encrpy_matrix_to_pair(self, W, scaling_factor, axis=1):
        if axis == 0:
            W = W.T
        EW0 = np.empty_like(W)
        EW1 = np.empty_like(W)

        for i, row in enumerate(W):
            for j, element in enumerate(row):
                E = self.esup.one_way_encrypt_vector([element], self.l, scaling_factor).astype('int64')
                EW0[i, j] = E[0]
                EW1[i, j] = E[1]

        return [EW0, EW1]

    def encrpy_vector_to_pair(self, V, scaling_factor):

        EV0 = np.empty_like(V)
        EV1 = np.empty_like(V)

        for j, element in enumerate(V):
                E = self.esup.one_way_encrypt_vector([element], self.l, scaling_factor).astype('int64')
                EV0[j] = E[0]
                EV1[j] = E[1]

        return [EV0, EV1]

    def s_decrypt(self, ee1):
        return self.esup.s_decrypt(ee1).astype('float')

    def decrpt_pair(self, ev_pair, scaling_factor):
        ## Input must be a pair of vector,
        ## will NOT work for matrices
        out = np.empty_like(ev_pair[0]).astype('float')
        e_out = np.stack(ev_pair, axis=-1)
        if e_out.ndim == 2:
            for i, e_pair in enumerate(e_out):
                out[i] = self.esup.s_decrypt(e_pair).astype('float') / scaling_factor

        return  out

    def encrypt_filtes(self, filters, scaling_factor, encrypt_direction=3):
        if encrypt_direction == 3:
            shape = (filters.shape[0], filters.shape[1], filters.shape[2], filters.shape[3]+1)
        elif encrypt_direction == 2:
            shape = (filters.shape[0], filters.shape[1], filters.shape[2]+1, filters.shape[3])
        encrypted_filters = np.zeros(shape)

        if encrypt_direction == 3:
            for k in range(filters.shape[2]):
                for i in range(filters.shape[0]):
                    for j in range(filters.shape[1]):
                        v = filters[i, j, k, :]
                        ev = self.one_way_encrypt_vector(v, scaling_factor)
                        encrypted_filters[i, j, k, :] = ev

        if encrypt_direction == 2:
            for k in range(filters.shape[3]):
                for i in range(filters.shape[0]):
                    for j in range(filters.shape[1]):
                        v = filters[i, j, :, k]
                        ev = self.one_way_encrypt_vector(v, scaling_factor)
                        encrypted_filters[i, j, :, k] = ev

        return encrypted_filters

    def dencrypt_filter_output(self, output, scaling_factor, ispair=False):

        dout = None

        if ispair:
            shape = (output.shape[0], output.shape[1], output.shape[2])
            dout = np.zeros(shape)
            for i in range(output.shape[0]):
                for j in range(output.shape[1]):
                    for k in range(output.shape[2]):
                        pair = [[output[i, j, k, 0]], [output[i, j, k, 1]]]
                        dout[i, j, k] = self.decrpt_pair(pair, scaling_factor)
        else:
            shape = (output.shape[0], output.shape[1], output.shape[2] - 1)
            dout = np.zeros(shape)
            for i in range(output.shape[0]):
                for j in range(output.shape[1]):
                    ev = output[i, j, :]
                    v = self.s_decrypt(ev)
                    dout[i, j, :] = v
            dout = dout / scaling_factor

        return dout

    def convolution2D(self, input, filters, isencrypted=False):
        padding = [(filters.shape[0] - 1)//2, (filters.shape[1] - 1)//2]
        image = np.zeros((2*padding[0]+input.shape[0], 2*padding[1]+input.shape[1], input.shape[2]))
        image[padding[0]:-padding[0], padding[1]:-padding[1], :] = input
        dim = 1
        if isencrypted:
            dim = 2

        output = np.zeros((input.shape[0], input.shape[1], filters.shape[3], dim))
        for k in range(output.shape[2]):
            for i in range(output.shape[0]):
                for j in range(output.shape[1]):
                    select = image[i+padding[0]-padding[0]:i+padding[0]+padding[0]+1, j+padding[1]-padding[1]:j+padding[1]+padding[1]+1, :]
                    filter = filters[:, :, :, k]
                    #output[i, j, k] = np.sum(np.multiply(filter, select))
                    out_temp = np.zeros(dim);
                    for ii in range(filter.shape[0]):
                        for jj in range(filter.shape[1]):
                            if isencrypted:
                                update = self.mat_mul_forward(select[ii, jj, :], [filter[ii, jj, :]], 1)
                            else:
                                update = np.dot(select[ii, jj, :], filter[ii, jj, :])
                            out_temp = out_temp + update

                        output[i, j, k, :] = out_temp
        return output

    def relu_vector(self, vector):
        h = np.zeros_like(vector)
        for k, v in enumerate(vector):
            if v > 0:
                h[k] = v
        return h

    def sum_pooling_2x2(self, matrix):
        out_shape = (matrix.shape[0]//2, matrix.shape[1]//2, matrix.shape[2])
        out_matrix = np.zeros(out_shape)
        for i in range(out_shape[0]):
            for j in range(out_shape[1]):
               out_matrix[i, j, :] = (matrix[2*i, 2*j, :] + matrix[2*i+1, 2*j, :] + matrix[2*i, 2*j+1, :] + matrix[2*i+1, 2*j+1, :]).reshape((out_matrix.shape[2],))
               #sum = matrix[2 * i, 2 * j, :] + matrix[2 * i + 1, 2 * j, :] + matrix[2 * i, 2 * j + 1,:] + matrix[2 * i + 1, 2 * j + 1, :]

        return out_matrix

    def relu_3Dmatrix(self, matrix):
        out_matrix = np.zeros_like(matrix)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                out_matrix[i, j, :] = self.relu_vector(matrix[i,j,:])
        return out_matrix

e = Encryption(2**8)