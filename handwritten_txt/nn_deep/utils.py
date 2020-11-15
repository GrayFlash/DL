import numpy as np
import matplotlib.pyplot as plt
import h5py

np.random.seed(1)

# Activation Functions
class utils:
    def sigmoid(Z):
        s = 1/(1+np.exp(-Z))
        cache = Z
        return s, cache

    def relu(Z):
        s = np.maximum(0, Z)
        cache = Z
        return s, cache

    # Differntials of Activation Functions

    def sigmoid_backward(dA, cache):
        Z = cache
        s = 1/(1+np.exp(-Z))
        dZ = dA*s*(1-s)
        
        return dZ

    def relu_backward(dA, cache):
        
        Z = cache
        dZ = np.array(dA, copy=True)
        dZ[Z<=0] = 0
        return dZ

    def initialize_parameters(layer_dims):
        
        L = len(layer_dims)
        parameters = {}
        np.random.seed(1)
        for l in range(1, L):
            parameters["W"+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])/np.sqrt(layer_dims[l-1])
            parameters["b"+str(l)] = np.zeros((layer_dims[l], 1))
            
        return parameters

    def linear_forward(A, W, b):
        
        Z = W.dot(A) + b
        cache = (A, W, b)
        
        return Z, cache

    def linear_forward_activation(A_prev, W, b, activation):
        
        if(activation=="sigmoid"):
            Z, linear_cache = linear_forward(A_prev, W, b)
            A, activation_cache = sigmoid(Z)
            
        
        elif(activation=="relu"):
            Z, linear_cache = linear_forward(A_prev, W, b)
            A, activation_cache = relu(Z)
            
        cache = (linear_cache, activation_cache)
        
        return A, cache

    def L_model_forward(X, parameters):
        caches = []
        A = X
        L = len(parameters)//2
        
        for l in range(1, L):
            A_prev = A
            A, cache = linear_forward_activation(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], 'relu')
            caches.append(cache)
        AL, cache = linear_forward_activation(A, parameters["W"+str(L)], parameters["b"+str(L)], 'sigmoid')
        
        caches.append(cache)
        
        return AL, caches

    def compute_cost(AL, Y):
        
        m = Y.shape[1]
        
        cost = (1.0/m) *( -np.dot(Y, np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
        cost = np.squeeze(cost)
        return cost

    def linear_backward(dZ, cache):
        A_prev, W, b = cache
        
        m = A_prev.shape[1]
        dW = 1./m * np.dot(dZ, A_prev.T)
        db = 1./m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        
        return dA_prev, dW, db

    def linear_backward_activation(dA, cache, activation):
        linear_cache, activation_cache = cache
        
        if(activation=='relu'):
            dZ = relu_backward(dA, activation_cache)
            dA_prev, dW, db = linear_backward(dZ, linear_cache)

        elif(activation=='sigmoid'):
            dZ = sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = linear_backward(dZ, linear_cache)
            
        
        return dA_prev, dW, db

    def L_model_backward(AL, Y, caches):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)
        
        dAL = - (np.divide(Y, AL) - np.divide(1-Y, 1-AL))
        
        current_cache = caches[L-1]
        grads["dA"+str(L-1)], grads["dW"+str(L)], grads["db"+str(L)] = linear_backward_activation(dAL, current_cache, "sigmoid")
        
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_t, dW_t, db_t = linear_backward_activation(grads["dA"+str(l+1)], current_cache, "relu")
            grads["dA"+str(l)] = dA_t
            grads["dW"+str(l+1)] = dW_t
            grads["db"+str(l+1)] = db_t
            
        return grads

    def update_parameters(parameters, grads, learning_rate):
        L = len(parameters) // 2
        
        for l in range(L):
            parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - learning_rate*grads["dW"+str(l+1)]
            parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - learning_rate*grads["db"+str(l+1)]
            
        return parameters

    def print_mislabeled_images(classes, X, y, p):
        a = p + y
        mislabeled_indices = np.asarray(np.where(a == 1))
        plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
        num_images = len(mislabeled_indices[0])
        for i in range(num_images):
            index = mislabeled_indices[1][i]
            
            plt.subplot(2, num_images, i + 1)
            plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
            plt.axis('off')
            plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))
