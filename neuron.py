
import numpy as np

class neural_network_operations():

    def relu(self, activation):#the activation function
        for ind,input in enumerate(activation):
            if input >= 0: #if positive then 1
                activation[ind] = 1
            else: #if negative then 0
                activation[ind] = 0
        return activation

    def weighted_sum(self, inputs, weights, bias):#sum of inputs*weights -> vector | works only for 1 node
        cur_activations = np.array([])
        if inputs.ndim == 1:
            if weights.ndim == 1:#a vector
                for input, weight in zip(inputs, weights):
                    cur_activations = np.append(cur_activations, input*weight)
            elif weights.ndim == 2:# a matrix
                input_activation = 0
                for weights in weights:
                    for input, weight in zip(inputs, weights):
                        input_activation += input*weight
                    cur_activations = np.append(cur_activations, input*weight)
            else:
                raise ValueError("Weights must be of type vector or 2D matrix")
        else:
            raise ValueError("inputs must be of type vector")
        cur_activations += bias
        
        return cur_activations


#weighted_sum(x, w, b)
x = np.array([1, 2, 3, 4])
w = np.array([-1, 0.4, 0.1, -0.8])
b = np.random.randn()* 0.01
neuron_ops = neural_network_operations()
weighted_sum = neuron_ops.weighted_sum(x, w, b)
activation = neuron_ops.relu(weighted_sum)
#print(activation)

x2 = np.array([3,6,7,8,9]) # 1x5
w2 = np.array([[0.3, 0.2, -0.5, -0.06,1],
              [0.99, 0.44, 0.66, -0.14, 0.52],
              [0.456, 0.113, 0.689, -0.957, 0.185],
              [0.68,-0.31,0.02,0.01,-0.24],
              [0.003,-0.008,-0.009,-0.003, 0.005]]) #5x5
print(neuron_ops.weighted_sum(x2,w2,b))