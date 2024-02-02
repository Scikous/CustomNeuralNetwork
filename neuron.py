import numpy as np
from PIL import Image, ImageOps
import glob
# from numba import jit, cuda
# from numba.experimental import jitclass
from timeit import default_timer as timer  
import os
import json
import concurrent.futures as cf

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
                sum_activation_probs += np.exp(activationn-np.max(output_layer_activations))
            probability = np.exp(activation-np.max(output_layer_activations))/sum_activation_probs
            softmax = np.append(softmax, probability)

        max_logit = np.max(output_layer_activations)
        exp_logits = np.exp(output_layer_activations - max_logit)
        softmax_values = exp_logits / np.sum(exp_logits)
        return softmax_values
    
    def sigmoid(self, output_layer_activations):
        sigmoid = np.array([])
        print(np.exp(-output_layer_activations[0]))
        for activation in output_layer_activations:
            probability =  0.0#
            #sum_activation_probs = 0.0            
            probability = np.exp(-activation)/(1+np.exp(-activation))**2
            sigmoid = np.append(sigmoid, probability)
        return sigmoid
    
    def feed_forward(self, inputs, num_hidden_layers:int, output_layer_size:int, weights=None, biases=None):
        weight_size = len(inputs)
        all_outputs = [np.array(inputs)]
        if weights == None:
            weights = [np.random.uniform(0.0, np.sqrt(2/len(inputs)), size=(len(inputs), weight_size)) for _ in range(num_hidden_layers)]
            output_layer_weights = np.random.uniform(0.0, np.sqrt(2/len(inputs)), size=(output_layer_size, weight_size))
            weights.append(output_layer_weights)
            biases = [np.random.uniform(0.0, np.sqrt(2/len(inputs)), size=(len(inputs))) for _ in range(num_hidden_layers)]
            output_layer_biases = np.random.uniform(0.0, np.sqrt(2/len(inputs)), size=(output_layer_size))
            biases.append(output_layer_biases)
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
    
    def cross_entropy_loss(self, output_layer, expected_layer):
        probabilities = output_layer
        epsilon = 1e-10
        if isinstance(expected_layer, (int, float)):
            loss = -(expected_layer*np.log(probabilities[0]+epsilon) +(1-expected_layer)*np.log(1-probabilities[0]+epsilon))
        else:
            loss = -(expected_layer*np.log(probabilities+epsilon) +(1-expected_layer)*np.log(1-probabilities+epsilon))
        return loss
    
    def cross_entropy_loss_backpropagation(self, output_layer, expected_layer):
        probabilities = output_layer
        epsilon = 1e-10

        if isinstance(expected_layer, (int, float)):
            loss = -(expected_layer/ (probabilities[0]+epsilon) - (1.0-expected_layer)/(1.0-probabilities[0]+epsilon))
        else:
            loss = -(expected_layer/(probabilities+epsilon) - (1.0-expected_layer)/(1.0-probabilities+epsilon))
        return loss
    
    def relu_backpropagation(self, activatables) -> 0 | 1:#derivative of relu
        activator = lambda activation: 0 if activation <= 0 else 1
        if isinstance(activatables, (int, float)):#if single neuron to activate
            return activator(activatables)
        else:
            activations = np.array([])
            for activatable in activatables:
                activations = np.append(activations, activator(activatable))
            return activations

    def softmax_backpropagation(self, activatables):
        return self.softmax(activatables)*(1-self.softmax(activatables))
        
    def sigmoid_backpropagation(self, activatables):
        return self.sigmoid(activatables)*(1-self.sigmoid(activatables))
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
        #neuron_mse_bps = np.array([])
        # for activation, expected in zip(output_layer_activations, expected_layer): #used for MSE
        #     neuron_mse_bps = np.append(neuron_mse_bps, self.mean_squared_error_backpropagation(activation, expected)) #MSE derivative
        costs_output_layer = self.cross_entropy_loss_backpropagation(output_layer_activations, expected_layer)
        softmax_activation_bps = self.softmax_backpropagation(output_layer_outputs) #relu derivative    
        errors = np.append(errors, self.hadamard_product(costs_output_layer, softmax_activation_bps))#element wise product
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
        # print("weights",len(gradient_weights), len(gradient_weights[0]), len(gradient_weights[-1]))
        # print(gradient_weights)
        #print("biases",len(gradient_biases), len(gradient_biases[0]), len(gradient_biases[-1]))
        #print(gradient_biases)   
        for ind, (weights, biases) in enumerate(zip(gradient_weights, gradient_biases)):#each batch element's set of weights and biases
            input_weights[ind] -= learning_rate*weights
            input_biases[ind] -= learning_rate*biases
        return input_weights, input_biases


    def backpropagation(self, all_outputs, output_layer_activations, expected_layer, all_weights):
        num_layers = len(all_weights) #num of weight vectors = num of layers
        #1st equation, get output layer errors
        output_layer_errors = self.output_layer_errors(all_outputs[-1], output_layer_activations, expected_layer)
        
        #2nd and 3rd equation, get individual layer errors (not including output layer) and all cost w.r.t bias values (is equal to layer error) 
        previous_layer_errors = self.layer_errors(all_weights[-1], output_layer_errors,  all_outputs[-2])#the layer before the output layer   
        all_cost_wrt_biases = [output_layer_errors, previous_layer_errors]
        for layer in range(num_layers-2, 0, -1):#not including ouput layer and the one before it, and the input layer
            layer_errors = self.layer_errors(all_weights[layer+1], previous_layer_errors,  all_outputs[layer])
            all_cost_wrt_biases.append(layer_errors)
            previous_layer_errors = layer_errors

        # #4th equation get the cost w.r.t weights
        all_cost_wrt_weights = []
        #activations * layer_errors of the previous layer
        all_outputs.pop(-1)
        for layer_errors, layer_outputs in zip(all_cost_wrt_biases, all_outputs[::-1]): #ignore output layer's activations
            cost_wrt_weights_in_neurons = []
            layer_outputs = self.relu(layer_outputs)
            for error in layer_errors:
                cost_wrt_weights_in_neuron = []
                for output in layer_outputs:    
                    cost_wrt_weights_in_neuron.append(output*error)
                cost_wrt_weights_in_neurons.append(cost_wrt_weights_in_neuron)
            all_cost_wrt_weights.append(np.array(cost_wrt_weights_in_neurons))

        return all_cost_wrt_weights[::-1], all_cost_wrt_biases[::-1]
    
    def batch_averages(self, batch_weights, batch_biases):#take the average of the weights and biases of the batch
        mean_weights = [np.zeros_like(batch_weights[0][0]) for _ in range(len(batch_weights[0])-1)]
        mean_output_weights = np.zeros_like(batch_weights[0][-1])#output layer can have different num of neurons
        mean_weights.append(mean_output_weights)

        mean_biases = [np.zeros_like(batch_biases[0][0]) for _ in range(len(batch_biases[0])-1)]
        mean_output_biases = np.zeros_like(batch_biases[0][-1])#output layer can have different num of neurons
        mean_biases.append(mean_output_biases)

        for weights_set, biases_set in zip(batch_weights, batch_biases):#each batch element's set of weights and biases
            for ind, (weights, biases) in enumerate(zip(weights_set, biases_set)):#for each layer sum weights with previous set of weights
                mean_weights[ind] += weights
                mean_biases[ind] += biases

        #after summation divide to get mean
        for ind in range(len(mean_biases)):
            mean_weights[ind] /= len(batch_weights)
            mean_biases[ind] /= len(batch_biases)



        #print(len(mean_weights), len(mean_weights[0]),len(mean_weights[-1]), len(mean_weights[0][0]), len(mean_weights[0][-1]))
        #print(len(mean_biases), len(mean_biases[0]), len(mean_biases[-1]))
        #print(mean_weights, mean_biases, len(mean_weights), len(mean_biases))
        return mean_weights, mean_biases
    
    def batch_trainer(self, train_path, batch_size, batch_iter, num_hidden_layers, learning_rate, grayscale=True):
        batch_train_imgs,train_ground_truths,train_cnt = images_loader(train_path, batch_size, grayscale=grayscale ,batch_iter=batch_iter)
        batch_weights = []
        batch_biases = []
        weights, biases =[], []
        for train_img, train_ground_truth in zip(batch_train_imgs, train_ground_truths):#for each train and test img, get new weights and biases
            num_classes = len(train_ground_truth)
            if weights and biases:
                all_outputs, output_layer_activations, weights, biases = self.feed_forward(train_img, num_hidden_layers, num_classes, weights, biases)
            else:
                all_outputs, output_layer_activations, weights, biases = self.feed_forward(train_img, num_hidden_layers, num_classes)
            #print("initial weights:",weights[0][:1])
            print('loss:', np.sum(self.cross_entropy_loss(output_layer_activations, train_ground_truth)))
            #print(output_layer_activations)
            gradient_weights, gradient_biases = self.backpropagation(all_outputs, output_layer_activations, train_ground_truth, weights)
            #print("gradient weights:",gradient_weights[0][:1])

            train_example_weights, train_example_biases = self.gradient_descent(weights, biases, gradient_weights, gradient_biases,  learning_rate)
            #print(weights[0][:1])
            batch_weights.append(train_example_weights)
            batch_biases.append(train_example_biases)
            #print("updated weights:",train_example_weights[0][:1])
            #print(len(batch_weights))  
            #           
        weights, biases = self.batch_averages(batch_weights, batch_biases)#calculate average values for all weights and biases
        return weights, biases

    def model_trainer(self, batch_size, epochs=1):
        testing_path = 'test'
        train_path = 'train'
        breaker = False        
        #batch_test_imgs,test_ground_truths,test_cnt = images_loader(testing_path, batch_size, grayscale=True)
        batch_train_imgs,train_ground_truths,train_cnt = images_loader(train_path, batch_size, grayscale=True)

        batch_iter = 0
        num_hidden_layers = 1
        learning_rate = 0.00001
        epoch_weights, epoch_biases = [], []
        
        while epochs > 0:
            print("Epochs Remaining:", epochs)
            while train_cnt > (batch_size*batch_iter) and batch_iter < 4:
                #weights, biases = self.batch_trainer(train_path, batch_size, batch_iter, num_hidden_layers, learning_rate)
                batch_iter += 1
                # epoch_weights.append(weights)
                # epoch_biases.append(biases)
                print("--------------------------------->",batch_iter)
            epoch_results = self.execute_batches_in_parallel([train_path, batch_size, batch_iter, num_hidden_layers, learning_rate])
            
            # data_names = ['weights', 'biases', 'hidden_layers', 'classes']
            # self.model_saver(data_names, weights, biases, num_hidden_layers)
            # if batch_iter > 5:
            #     break
            epochs -= 1

        # weights,biases = self.batch_averages(epoch_weights, epoch_biases)
        print(len(epoch_results))
        # return weights, biases, num_hidden_layers, train_ground_truths[0]
    
    def execute_batches_in_parallel(self, data):
        train_path, batch_size, batch_iter, num_hidden_layers, learning_rate = data
        with cf.ProcessPoolExecutor() as executor:
            res = [executor.submit(self.batch_trainer, train_path, batch_size, batch_iter, num_hidden_layers, learning_rate) for batch_iter in range (batch_iter)]
            cf.wait(res)
        return res
    
    def model_saver(self, data_names:list, *args):
        subdirectories = [d for d in os.listdir('train') if os.path.isdir(os.path.join('train', d))]
        #label_mapping = {label: index for index, label in enumerate(subdirectories)}
        model_file_path = "model_save_state.json"
        data_to_save = {}
        for name,arg in zip(data_names, args):
            if name != 'hidden_layers':
                arg = [arr.tolist() for arr in arg]
            data_to_save[name] = arg

        data_to_save['classes'] = subdirectories

        #print(data_to_save)
        with open(model_file_path, 'w') as json_file:
            json.dump(data_to_save, json_file)

    def model_loader(self):
        model_file_path = "model_save_state.json"
                
        with open(model_file_path, 'r') as json_file:
            data = json.load(json_file)
        weights = [np.array(saved_weights) for saved_weights in data["weights"]]
        biases = [np.array(saved_biases) for saved_biases in data["biases"]]
        num_hidden_layers = data["hidden_layers"]
        classes = data["classes"]

        return weights, biases, num_hidden_layers, classes          
    

    
def images_loader(dir_path, batch_size, grayscale=False, batch_iter=0):#loads a batch of images to be used

    images_vectorized = np.array([])
    subdirectories = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    #label_mapping = {label: index for index, label in enumerate(subdirectories)}
    #print(label_mapping)
    batch_leftover = batch_size % len(subdirectories)
    batch_size = (batch_size-batch_leftover)//len(subdirectories)
    batch_take_from = batch_iter*batch_size
    batch_take_until = batch_take_from+batch_size 

    print(batch_take_from,batch_take_until, batch_size)
    ground_truths = []
    #print(subdirectories)
    train_cnt = 0
    for subdir in subdirectories:#loop through directories of classes and get labels and vectorized versin of all imgs
        images_path = glob.glob(dir_path+'/'+subdir+'/*.jpg') + glob.glob(dir_path+'/'+subdir+'/*.png')
        np.random.shuffle(images_path)#shuffle dataset to avoid getting stuck
        if train_cnt == 0:
            train_cnt = len(images_path)
        #print(images_path[batch_take_from:batch_take_until])
        if grayscale:
            images_arr = np.array([ImageOps.grayscale(Image.open(img_path).resize((35,35))) for img_path in images_path[batch_take_from:batch_take_until]])
        else:
            images_arr = np.array([Image.open(img_path) for img_path in images_path[batch_take_from:batch_take_until ]])
        images_arr = np.array([img.reshape(-1)/255.0 for img in images_arr])
        label = [0] * len(subdirectories)
        label[subdirectories.index(subdir)] = 1
        ground_truths.append([label]*(batch_size//2))#multiple label for each image in class
        images_vectorized = np.vstack([images_vectorized, images_arr]) if images_vectorized.size else images_arr

    #print(ground_truths)
    ground_truths = np.concatenate(np.array(ground_truths), axis=0)#collapse set of classes to singular set of classes
    #print(images_vectorized,'\n', ground_truths)

    #shuffle data while keeping labels and images one-to-one
    shuffled_indices = np.random.permutation(batch_size)
    ground_truths = ground_truths[shuffled_indices]
    images_vectorized = images_vectorized[shuffled_indices]
    #print(images_vectorized)
    return images_vectorized, ground_truths, train_cnt-batch_take_until


if __name__=="__main__":
    neuron_ops = neural_network_operations()

    testing_path = 'test'
    train_path = 'train'
    batch_size = 9

    # weights, input_biases, num_hidden_layers, classes = neuron_ops.model_loader()
    # img = np.array([ImageOps.grayscale(Image.open('valid/cat_9999.png').resize((35,35)))])
    # img = np.array([img.reshape(-1)/255.0])
    # ground_truth = np.array([1, 0, 0])
    # print(classes)
    # print(img)

    # all_outputs, output_layer_activations, weights, biases = neuron_ops.feed_forward(img[0], num_hidden_layers, len(classes), weights, input_biases)
    # print(output_layer_activations)

    neuron_ops.model_trainer(batch_size, 2)
    #execute_batches_in_parallel(fun, batch_size)



        #print(weights,'\n\n', input_biases)


    # x = np.array([1, 2, 3, 4])
    # w = np.array([-1, 0.4, 0.1, -0.8])

    # b = np.random.randn()* 0.01
    # weighted_sum = neuron_ops.weighted_sum_output(x, w, b)
    # #print(weighted_sum)
    # activation = neuron_ops.relu(weighted_sum)
    # #print(activation)

    # x2 = np.array([3,6,7,8,9]) # 1x5
    # w2 = np.array([[0.3, 0.2, -0.5, -0.06,1],
    #             [0.99, 0.44, 0.66, -0.14, 0.52],
    #             [0.456, 0.113, 0.689, -0.957, 0.185],
    #             [0.68,-0.31,0.02,0.01,-0.24],
    #             [0.003,-0.008,-0.009,-0.003, 0.005]]) #5x5
    # hidden_layer_activations = neuron_ops.weighted_sum_layer(x2,w2,b)
    # output_layer = neuron_ops.weighted_sum_output(hidden_layer_activations, w, -0.2)
    # relu_activated = neuron_ops.relu(output_layer)
    # #print(hidden_layer_activations, output_layer, relu_activated)

    # E = np.array([1,0,1])
    # hidden_layer_activations = np.array([3, 6, 8])
    # #print(hidden_layer_activations)
    # all_outputs, output_layer_activations, weights, biases = neuron_ops.feed_forward(hidden_layer_activations, 2, 3)
    # #print(outputL)
    # #print(neuron_ops.backpropagation(outputL,E, weights, biases))
    # loss = neuron_ops.mean_squared_error(output_layer_activations, E)
    # print(loss)
    # gradient_weights, gradient_biases = neuron_ops.backpropagation(all_outputs, output_layer_activations, E, weights, biases)

    # input_weights, input_biases = neuron_ops.gradient_descent(weights, biases, gradient_weights, gradient_biases,  0.001)


    # all_outputs, output_layer_activations, weights, biases = neuron_ops.feed_forward(hidden_layer_activations, 2, 2, input_weights, input_biases)
    # loss = neuron_ops.cross_entropy_loss(output_layer_activations, E)#neuron_ops.mean_squared_error(output_layer_activations, E)
    # loss2 = neuron_ops.cross_entropy_loss_backpropagation(output_layer_activations, E)#neuron_ops.mean_squared_error(output_layer_activations, E)

    # print('\n\nhello', loss, loss2)
    # #print(input_weights,'\n\n', input_biases)