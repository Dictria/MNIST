import numpy as np
from layer_utils import sigmoid, sigmoid_backward, ReLU, ReLU_backward


class Model:
    def __init__(self, layer_dims, num_classes, activation, reg=0.0):
        self.parameters = self.initialize_parameters(layer_dims)
        self.reg = reg
        self.num_classes = num_classes
        self.activation = activation

    def initialize_parameters(self, layer_dims):
        '''
        :param layer_dims: list containing the dimensions of each layer in our network
        :return: dictionary containing parameters "W1", "b1", ...
        '''
        parameters = {}
        L = len(layer_dims)
        for i in range(1, L):
            parameters['W' + str(i)] = np.random.normal(0, 2 / (layer_dims[i - 1] + layer_dims[i]),
                                                        (layer_dims[i], layer_dims[i - 1]))
            parameters['b' + str(i)] = np.zeros((layer_dims[i], 1))

        #for k, v in parameters.items():
        #    print(f'{k} shape: {v.shape}')
        return parameters

    def linear_forward(self, A, W, b):
        #print(f'linear_forward--A: {A.shape}, W: {W.shape}')
        Z = np.dot(W, A) + b
        cache = (A, W, b)
        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation):
        '''
        :param activation: 'sigmoid' or 'ReLU'
        :return:
        '''
        if activation != 'sigmoid' and activation != 'ReLU':
            raise Exception(f'Parameter error: linear_activation_forward has no parameter values {activation}')
        Z, linear_cache = self.linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z) if activation == 'sigmoid' else ReLU(Z)
        cache = (linear_cache, activation_cache)
        return A, cache

    def forwawrd(self, X):
        caches = []
        A = X
        L = len(self.parameters) // 2
        for i in range(1, L):
            A_prev = A
            A, cache = self.linear_activation_forward(A_prev, self.parameters['W' + str(i)],
                                                      self.parameters['b' + str(i)], self.activation[i - 1])
            caches.append(cache)
        AL, cache = self.linear_forward(A, self.parameters['W' + str(L)], self.parameters['b' + str(L)])
        caches.append(cache)
        return AL, caches

    def softmax_loss(self, AL, y):
        loss, dx = None, None
        tmp = AL.T
        num_train = tmp.shape[0]
        scores = tmp - np.max(tmp, axis=1).reshape(num_train, 1)
        p = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(num_train, 1)
        loss = -1 * np.sum(np.log(p[np.arange(num_train), y])) / num_train
        p[np.arange(num_train), y] -= 1
        dA = p / num_train
        return loss, dA.T

    def linear_backward(self, dZ, cache):
        '''
        :param cache: tuple of values (A_prev, W, b)
        '''
        A_prev, W, b = cache
        #print(f'dZ: {dZ.shape}, A_prev: {A_prev.shape}')
        dW = dZ.dot(A_prev.T)
        db = np.sum(dZ, axis=1, keepdims=True)
        #print(f'db: {db.shape}')
        dA_prev = W.T.dot(dZ)
        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):
        linear_cache, activation_cache = cache
        if activation == 'ReLU':
            dZ = ReLU_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        elif activation == 'sigmoid':
            dZ = sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        return dA_prev, dW, db

    def backward(self, AL, Y, caches):
        L = len(caches)  # the number of layers
        grads = {}

        loss, dAL = self.softmax_loss(AL, Y)
        for i in range(1, L+1):
            loss += 0.5 * self.reg * np.sum(np.square(self.parameters['W' + str(i)]))

        current_cache = caches[L - 1]
        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = \
            self.linear_backward(dAL, current_cache)
        grads["dW" + str(L)] += self.reg * self.parameters['W' + str(L)]

        for i in reversed(range(L - 1)):
            current_cache = caches[i]
            dA_prev_tmp, dW_tmp, db_tmp = \
                self.linear_activation_backward(grads["dA" + str(i + 2)], current_cache, self.activation[i])
            grads["dA" + str(i + 1)] = dA_prev_tmp
            grads["dW" + str(i + 1)] = dW_tmp + self.reg * self.parameters['W' + str(i + 1)]
            grads["db" + str(i + 1)] = db_tmp
        return loss, grads

    def update_parameters(self, grads, learning_rate):
        L = len(self.parameters) // 2
        for i in range(L):
            self.parameters['W' + str(i + 1)] = self.parameters['W' + str(i + 1)] - learning_rate * grads[
                'dW' + str(i + 1)]
            self.parameters['b' + str(i + 1)] = self.parameters['b' + str(i + 1)] - learning_rate * grads[
                'db' + str(i + 1)]
        return self.parameters

    def step(self):
        pass

    def train(self):
        pass

    def loss(self, X, Y=None):
        score, caches = self.forwawrd(X)
        if Y is None:
            return score
        loss, grads = self.backward(score, Y, caches)
        return loss, grads
