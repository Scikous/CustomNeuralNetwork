import numpy as np

class neural_network_operations():

    def relu(self, activatables):#the activation function
        if isinstance(activatables, (int, float)):#if single neuron to activate
            return max(0,activatables)
        else:
            activations = np.array([])
            for activatable in activatables:
                activations = np.append(activations, max(0,activatable))
            return activations


    def weighted_sum_output(self, inputs, weights, bias):#single neuron output, sum of inputs*weights -> scalar
        if inputs.ndim == 1:#inputs are of type vector
            output = 0
            if weights.ndim == 1:#assuming one neuron per layer
                for input, weight in zip(inputs, weights):
                    output += input*weight
                output += bias
            else:
                raise ValueError("Weights must be of type vector")
        else:
            raise ValueError("inputs must be of type vector")
        return output

    def weighted_sum_layer(self, inputs, weights, bias):#for multiple neurons in layer (could be hidden layer or output layuer)
        outputs = np.array([])
        if weights.ndim == 2:#assuming multiple neurons per layer
            for weight in weights:#each row consists of the cur layer's weights
                neuron_output = self.weighted_sum_output(inputs, weight, 0)
                outputs = np.append(outputs, neuron_output)
        else:
            raise ValueError("Weights must be 2D matrix")
        outputs += bias
        return outputs

    def feed_forward(self, inputs, num_hidden_layers:int, multi_output=False, weights=None, biases=None):
        weight_size = len(inputs)
        if multi_output:#output layer is also calculated in while loop
            num_hidden_layers += 1
        if weights == None:
            weights = np.array([np.random.uniform(-1, 1, size=(len(inputs), weight_size)) for _ in range(num_hidden_layers)])
            biases = np.random.uniform(np.random.randn() * 0.01, size=(len(inputs)))
        
        ind = 0
        while num_hidden_layers > 0:#0 = output layer
            hidden_layer_outputs = self.weighted_sum_layer(inputs, weights[ind], biases[ind])
            hidden_layer_outputs = self.relu(hidden_layer_outputs)
            inputs = hidden_layer_outputs
            num_hidden_layers -= 1
            ind += 1
        
        if not multi_output:
            weights = np.random.uniform(-1, 1, size=len(inputs))
            bias = np.random.randn() * 0.01
            output = self.weighted_sum_output(inputs, weights, bias)
            return self.relu(output)        
        return inputs, weights, biases

    def mean_squared_error(self, output_layer, expected_layer):
        if isinstance(output_layer, (int, float)):#if only one output in output_layer
            return (output_layer-expected_layer)**2
        else:
            sum = 0.0
            for output,expected in zip(output_layer, expected_layer):
                sum += (output-expected)**2

        M_S_E = sum/len(output_layer)
        return M_S_E
    
    def mean_squared_error_backpropagation(self, output_layer, expected_layer):
        if isinstance(output_layer, (int, float)):#if only one output in output_layer
            return 2*(output_layer-expected_layer)
        else:
            sum = 0.0
            for output,expected in zip(output_layer, expected_layer):
                sum += (output-expected)

        M_S_E_bp = 2*sum/len(output_layer)
        return M_S_E_bp
    
    def relu_backpropagation(self, activatables) -> 0 | 1:#derivative of relu
        activations = np.array([])
        activator = lambda activation: 0 if activation <= 0 else 1
        if isinstance(activatables, (int, float)):#if single neuron to activate
            return activator(activatables)
        else:
            for activatable in activatables:
                activations = np.append(activations, activator(activatable))
            return activations
        
    def hadamard_product(self, cost_gradient, activations_backpropagation):
        if isinstance(cost_gradient, (int, float)):#if only one gradient value
            return cost_gradient*activations_backpropagation
        else:
            hadamard_res = np.array([])
            for cost, activation in zip(cost_gradient, activations_backpropagation):
                hadamard_res = np.append(hadamard_res, cost*activation)
        return hadamard_res

    def output_layer_errors(self, output_layer, expected_layer) -> np.array([]):#gets the last layer's errors
        errors = np.array([])
        for output, expected in zip(output_layer, expected_layer):
            neuron_mse_bp = self.mean_squared_error_backpropagation(output, expected) #MSE derivative
            relu_activation_bp = self.relu_backpropagation(output) #relu derivative
            errors = np.append(errors, self.hadamard_product(neuron_mse_bp, relu_activation_bp))#element wise product
        return errors

    def backpropagation(self, output_layer, expected_layer, weights, biases):
        output_layer_errors = self.output_layer_errors(output_layer, expected_layer)

        return output_layer_errors


# t = np.array([[1, 0, 1],
#               [0,1,0],
#               [1,1,1]])

# t2 = np.array([[0, 0, 0],
#               [1,1,1],
#               [0,0,1]])


# print(np.sum(t-t2))

#weighted_sum(x, w, b)
x = np.array([1, 2, 3, 4])
w = np.array([-1, 0.4, 0.1, -0.8])

b = np.random.randn()* 0.01
neuron_ops = neural_network_operations()
weighted_sum = neuron_ops.weighted_sum_output(x, w, b)
print(weighted_sum)
activation = neuron_ops.relu(weighted_sum)
print(activation)

x2 = np.array([3,6,7,8,9]) # 1x5
w2 = np.array([[0.3, 0.2, -0.5, -0.06,1],
              [0.99, 0.44, 0.66, -0.14, 0.52],
              [0.456, 0.113, 0.689, -0.957, 0.185],
              [0.68,-0.31,0.02,0.01,-0.24],
              [0.003,-0.008,-0.009,-0.003, 0.005]]) #5x5
hidden_layer_activations = neuron_ops.weighted_sum_layer(x2,w2,b)
output_layer = neuron_ops.weighted_sum_output(hidden_layer_activations, w, -0.2)
relu_activated = neuron_ops.relu(output_layer)
print(hidden_layer_activations, output_layer, relu_activated)

E = np.array([1,0,1, 0, 1])

outputL, weights, biases = neuron_ops.feed_forward(hidden_layer_activations, 3, True)
print(outputL)
print(neuron_ops.backpropagation(outputL,E, weights, biases))
#outputL_errors = neuron_ops.output_layer_errors(outputL, E)
#print(outputL_errors)

