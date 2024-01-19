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

    def weighted_sum_layer(self, inputs, weights, biases):#for multiple neurons in layer (could be hidden layer or output layuer)
        outputs = np.array([])
        if weights.ndim == 2:#assuming multiple neurons per layer
            for weight in weights:#each row consists of the cur layer's weights
                neuron_output = self.weighted_sum_output(inputs, weight, 0)
                outputs = np.append(outputs, neuron_output)
        else:
            raise ValueError("Weights must be 2D matrix")
        outputs += biases
        return outputs

    def softmax(self, output_layer_activations):
        softmax = np.array([])
        for activation in output_layer_activations:
            probability =  0.0#
            sum_activation_probs = 0.0            
            for activationn in output_layer_activations:
                sum_activation_probs += np.exp(activationn)
            probability = np.exp(activation)/sum_activation_probs
            softmax = np.append(softmax, probability)
        print(softmax, output_layer_activations)
        return softmax
    
    def feed_forward(self, inputs, num_hidden_layers:int, output_layer_size:int, weights=None, biases=None):
        weight_size = len(inputs)
        all_outputs = [np.array(inputs)]
        if weights == None:
            weights = [np.random.uniform(-1, 1, size=(len(inputs), weight_size)) for _ in range(num_hidden_layers)] #account for input_layer
            output_layer_weights = np.random.uniform(-1, 1, size=(output_layer_size, weight_size))
            weights.append(output_layer_weights)
            biases = np.random.uniform(np.random.randn() * 0.01, size=(num_hidden_layers+1))
        
        ind = 0
        #pass layer activations through all hidden layers and calculates activations
        while num_hidden_layers > 0:#0 = output layer
            hidden_layer_outputs = self.weighted_sum_layer(inputs, weights[ind], biases[ind])
            all_outputs.append(hidden_layer_outputs) #size = 1xnum_hidden_layers
            hidden_layer_outputs = self.relu(hidden_layer_outputs)
            inputs = hidden_layer_outputs

            num_hidden_layers -= 1
            ind += 1  
        
        output_layer_outputs = self.weighted_sum_layer(inputs, weights[ind], biases[ind])
        all_outputs.append(output_layer_outputs)
        output_layer_activations = self.softmax(output_layer_outputs)
        return all_outputs, output_layer_activations, weights, biases

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

    def softmax_backpropagation(self, activatables):
        #activations = np.array([])
        #activator = lambda activation: self.softmax(activation)*(1-self.softmax(activation))
        # if isinstance(activatables, (int, float)):#if single neuron to activate
        #     return activator(activatables)
        #activations = np.append(activations, activator(activatables))
        return self.softmax(activatables)*(1-self.softmax(activatables))
        
    def hadamard_product(self, cost_gradient, activations_backpropagation):
        if isinstance(cost_gradient, (int, float)):#if only one gradient value
            return cost_gradient*activations_backpropagation
        else:
            hadamard_res = np.array([])
            for cost, activation in zip(cost_gradient, activations_backpropagation):
                hadamard_res = np.append(hadamard_res, cost*activation)
        return hadamard_res

    def output_layer_errors(self, output_layer_outputs, output_layer_activations, expected_layer) -> np.array([]):#gets the last layer's errors
        errors = np.array([])
        neuron_mse_bps = np.array([])
        print(output_layer_outputs, output_layer_activations)
        for activation, expected in zip(output_layer_activations, expected_layer):
            neuron_mse_bps = np.append(neuron_mse_bps, self.mean_squared_error_backpropagation(activation, expected)) #MSE derivative
        softmax_activation_bps = self.softmax_backpropagation(output_layer_outputs) #relu derivative
        errors = np.append(errors, self.hadamard_product(neuron_mse_bps, softmax_activation_bps))#element wise product
        return errors
    
    def layer_errors(self, previous_layer_weights, previous_layer_errors, current_layer_outputs):
        layer_errors = [] 
        weighted_sum_errors = []
        for weights in np.transpose(previous_layer_weights):
            weighted_sum_error = 0.0 
            for weight, error in zip(weights, previous_layer_errors):
                weighted_sum_error += weight * error
            weighted_sum_errors.append(weighted_sum_error)
            weighted_sum_error = 0.0

        layer_errors = self.hadamard_product(weighted_sum_errors, self.relu_backpropagation(current_layer_outputs))
        return layer_errors
    
    def gradient_descent(self, input_weights, input_biases, gradient_weights,  gradient_biases, learning_rate):
        for i in range(len(gradient_weights)):            
            for j  in range(len(gradient_weights[i])):
                #print(gradient_weights[i][j])
                for k in range(len(gradient_weights[i][j])):
                    #print(gradient_weights[i][j][k])
                    gradient_weights[i][j][k] = gradient_weights[i][j][k]* learning_rate
                    input_weights[i][j][k] -= gradient_weights[i][j][k]

        for i in range(len(gradient_biases)):            
            for j  in range(len(gradient_biases[i])):
                gradient_biases[i][j] = input_biases[i] - learning_rate*gradient_biases[i][j]
        input_biases = gradient_biases
        return input_weights, input_biases


    def backpropagation(self, all_outputs, output_layer_activations, expected_layer, all_weights, all_biases):
        num_layers = len(all_weights) #num of weight vectors = num of layers
        #print(all_weights)
        #1st equation, get output layer errors
        output_layer_errors = self.output_layer_errors(all_outputs[-1], output_layer_activations, expected_layer)
        
        #2nd and 3rd equation, get individual layer errors (not including output layer) and all cost w.r.t bias values (is equal to layer error) 
        previous_layer_errors = self.layer_errors(all_weights[-1], output_layer_errors,  all_outputs[-2])#the layer before the output layer   
        all_cost_wrt_biases = [output_layer_errors, previous_layer_errors]
        for layer in range(num_layers-2, 0, -1):#not including ouput layer and the one before it, and the input layer
            print(layer)
            #if layer ==  
            layer_errors = self.layer_errors(all_weights[layer+1], previous_layer_errors,  all_outputs[layer])
            all_cost_wrt_biases.append(layer_errors)
            previous_layer_errors = layer_errors

        
        # #4th equation get the cost w.r.t weights
        all_cost_wrt_weights = []
        #layer_errors * activations of the previous layer
        for layer_errors, layer_outputs in zip(all_cost_wrt_biases, all_outputs[-2::-1]): #ignore output layer's activations
            cost_wrt_weights_in_neurons = []
            layer_outputs = self.relu(layer_outputs)
            for error in layer_errors:
                cost_wrt_weights_in_neuron = []
                for output in layer_outputs:    
                    cost_wrt_weights_in_neuron.append(error*output)
                cost_wrt_weights_in_neurons.append(cost_wrt_weights_in_neuron)
            all_cost_wrt_weights.append(cost_wrt_weights_in_neurons)
        return all_cost_wrt_weights[::-1], all_cost_wrt_biases[::-1]


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
#print(weighted_sum)
activation = neuron_ops.relu(weighted_sum)
#print(activation)

x2 = np.array([3,6,7,8,9]) # 1x5
w2 = np.array([[0.3, 0.2, -0.5, -0.06,1],
              [0.99, 0.44, 0.66, -0.14, 0.52],
              [0.456, 0.113, 0.689, -0.957, 0.185],
              [0.68,-0.31,0.02,0.01,-0.24],
              [0.003,-0.008,-0.009,-0.003, 0.005]]) #5x5
hidden_layer_activations = neuron_ops.weighted_sum_layer(x2,w2,b)
output_layer = neuron_ops.weighted_sum_output(hidden_layer_activations, w, -0.2)
relu_activated = neuron_ops.relu(output_layer)
#print(hidden_layer_activations, output_layer, relu_activated)

E = np.array([1,0])
hidden_layer_activations = np.array([3, 6, 8])
#print(hidden_layer_activations)
all_outputs, output_layer_activations, weights, biases = neuron_ops.feed_forward(hidden_layer_activations, 2, 2)
#print(outputL)
#print(neuron_ops.backpropagation(outputL,E, weights, biases))
loss = neuron_ops.mean_squared_error(output_layer_activations, E)
print(loss)
gradient_weights, gradient_biases = neuron_ops.backpropagation(all_outputs, output_layer_activations, E, weights, biases)

input_weights, input_biases = neuron_ops.gradient_descent(weights, biases, gradient_weights, gradient_biases,  0.001)


all_outputs, output_layer_activations, weights, biases = neuron_ops.feed_forward(hidden_layer_activations, 2, 2, input_weights, input_biases)
loss = neuron_ops.mean_squared_error(output_layer_activations, E)
print(loss)
#print(input_weights,'\n\n', input_biases)
