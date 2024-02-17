import numpy as np
from PIL import Image, ImageOps
import glob
# from numba import jit, cuda
# from numba.experimental import jitclass
from timeit import default_timer as timer  
import os
import json
import concurrent.futures as cf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

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
    
    def feed_forward(self, inputs, num_hidden_layers:int, output_layer_size:int, weights=None, biases=None, weight_size=128):
        all_outputs = [np.array(inputs)]
        if weights == None:
            first_hidden_layer_weights = np.random.uniform(0.0, np.sqrt(2/len(inputs)), size=(weight_size, len(inputs)))#first hidden takes in the input features
            weights = [np.random.uniform(0.0, np.sqrt(2/len(inputs)), size=(weight_size, weight_size)) for _ in range(num_hidden_layers-1)] #ignore first hidden layer weights
            weights.insert(0, first_hidden_layer_weights)
            output_layer_weights = np.random.uniform(0.0, np.sqrt(2/len(inputs)), size=(output_layer_size, weight_size))
            weights.append(output_layer_weights)
            biases = [np.random.uniform(0.0, np.sqrt(2/len(inputs)), size=(weight_size)) for _ in range(num_hidden_layers)]
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

    def output_layer_errors(self, output_layer_outputs: np.array, output_layer_activations: np.array, expected_layer: np.array) -> np.array:#gets the last layer's errors
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
        #print("Gradient: ",gradient_weights[0].shape, len(gradient_weights), len(gradient_biases), gradient_biases[0].shape)
       # print("\ninput: ", input_weights[0].shape, len(input_weights), len(input_biases), input_biases[0].shape)
        #print("biases",len(gradient_biases), len(gradient_biases[0]), len(gradient_biases[-1]))
        #print(gradient_biases)   
        for ind, (weights, biases) in enumerate(zip(gradient_weights, gradient_biases)):#each batch element's set of weights and biases
            #print("\nIND:", ind)
            #print("weight shapes", input_weights[1].shape, gradient_weights[1].shape)
            input_weights[ind] -= learning_rate*weights
            #print("weightss", input_weights[ind].shape, weights.shape)
            input_biases[ind] -= learning_rate*biases
            #print("biases", input_biases[ind].shape, biases.shape)

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
        mean_weights_first_hidden = np.zeros_like(batch_weights[0][0])
        mean_weights = [np.zeros_like(batch_weights[0][1]) for _ in range(len(batch_weights[0])-2)]#first and output layer can have different neuron sizes
        mean_output_weights = np.zeros_like(batch_weights[0][-1])#output layer can have different num of neurons
        mean_weights.append(mean_output_weights)
        mean_weights.insert(0, mean_weights_first_hidden)
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
    
    def batch_trainer(self, batch_train_imgs, batch_train_labels, weights, biases, num_classes, num_hidden_layers, learning_rate, grayscale=True):

        #batch_train_imgs,batch_train_labels = images_loader(train_path, batch_size, grayscale=grayscale , batch_num=batch_num)
        batch_weights = []
        batch_biases = []
        average_loss = 0.0
        for train_img, batch_train_label in zip(batch_train_imgs, batch_train_labels):#for each train and test img, get new weights and biases
            if weights and biases:
                all_outputs, output_layer_activations, weights, biases = self.feed_forward(train_img, num_hidden_layers, num_classes, weights, biases)
            else:
                all_outputs, output_layer_activations, weights, biases = self.feed_forward(train_img, num_hidden_layers, num_classes)
            #print("initial weights:",weights[0][:1])
            average_loss += np.sum(self.cross_entropy_loss(output_layer_activations, batch_train_label))
            #print('loss:', np.sum(self.cross_entropy_loss(output_layer_activations, batch_train_label)))
            #print(len(train_img))
            gradient_weights, gradient_biases = self.backpropagation(all_outputs, output_layer_activations, batch_train_label, weights)
            #print("gradient weights:",gradient_weights[0][:1])

            train_example_weights, train_example_biases = self.gradient_descent(weights, biases, gradient_weights, gradient_biases,  learning_rate)
            #print(weights[0][:1])
            batch_weights.append(train_example_weights)
            batch_biases.append(train_example_biases)
            
            #print("updated weights:",train_example_weights[0][:1])
            #print(len(batch_weights))  
            #           
        weights, biases = self.batch_averages(batch_weights, batch_biases)#calculate average values for all weights and biases
        average_loss /= len(batch_train_imgs)
        return weights, biases, average_loss

    def get_num_batches(self, batch_train_imgs, batch_size):
        #subdirectories = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
        num_batches = 0
        # for subdir in subdirectories:
        #     images = glob.glob(train_path+'/'+subdir+'/*.jpg') + glob.glob(train_path+'/'+subdir+'/*.png')
        #     if train_cnt == 0 or train_cnt > len(images):#final batch of images is determined by the dir with the least num of images
        #         train_cnt = len(images)
        while len(batch_train_imgs) > (batch_size*num_batches):#batch traun_imgs is a vector consisting of images
                #weights, biases = self.batch_trainer(train_path, batch_size, num_batches, num_hidden_layers, learning_rate)
                num_batches += 1
        return num_batches
    def model_trainer(self, batch_size, epochs=1):
        testing_path = 'test'
        train_path = 'train'
        breaker = False        
        #batch_test_imgs,test_ground_truths,test_cnt = images_loader(testing_path, batch_size, grayscale=True)  
        (train_imgs,train_labels) , (_, _) = mnist.load_data()
        train_labels = to_categorical(train_labels)
        num_batches = self.get_num_batches(train_imgs, batch_size)
        num_classes = len(train_labels[0])
        print(num_classes)
        num_hidden_layers = 2
        learning_rate = 1e-3
        final_weights, final_biases = [], []
        lowest_loss = -1
        print(num_batches, num_classes)
        while epochs > 0:
            average_epoch_loss = []
            epoch_weights, epoch_biases = [], []
            epoch_train_imgs_batches, epoch_train_labels_batches = [], []
            print("Epochs Remaining:", epochs)
                # epoch_weights.append(weights)
                # epoch_biases.append(biases)
            for batch_num in range(num_batches):
                batch_train_imgs,batch_train_labels = mnist_batches(train_imgs, train_labels, batch_size, batch_num)
                epoch_train_imgs_batches.append(batch_train_imgs)
                epoch_train_labels_batches.append(batch_train_labels)
            epoch_results = self.execute_batches_in_parallel([epoch_train_imgs_batches, epoch_train_labels_batches, final_weights, final_biases, num_classes, num_hidden_layers, learning_rate])
            
            for batch_results in epoch_results:
                epoch_weights.append(batch_results[0])
                epoch_biases.append(batch_results[1])
                average_epoch_loss.append(batch_results[2])
            final_weights, final_biases = self.batch_averages(epoch_weights, epoch_biases)

            average_epoch_loss = np.sum(average_epoch_loss)/num_batches
            print('Epoch loss:',average_epoch_loss, lowest_loss)
            if average_epoch_loss < lowest_loss or lowest_loss == -1:
                data_names = ['weights', 'biases', 'hidden_layers', 'classes']
                self.model_saver(data_names, final_weights, final_biases, num_hidden_layers)
                lowest_loss = average_epoch_loss
            epochs -= 1

        # weights,biases = self.batch_averages(epoch_weights, epoch_biases)
            
        print(len(final_weights))
        # return weights, biases, num_hidden_layers, batch_train_labels[0]
    
    def execute_batches_in_parallel(self, data):
        train_imgs_batches, train_labels_batches, weights, biases, num_classes, num_hidden_layers, learning_rate = data
        with cf.ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.batch_trainer, batch_train_imgs, batch_train_labels, weights, biases, num_classes, num_hidden_layers, learning_rate) for (batch_train_imgs, batch_train_labels) in zip(train_imgs_batches, train_labels_batches)]
            cf.wait(futures)
        results = [future.result() for future in cf.as_completed(futures)]

        return results
    
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
    

    
def images_loader(dir_path, batch_size, grayscale=False, batch_num=0):#loads a batch of images to be used

    images_vectorized = np.array([])
    subdirectories = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    #label_mapping = {label: index for index, label in enumerate(subdirectories)}
    #print(label_mapping)
    batch_leftover = batch_size % len(subdirectories)
    batch_size = (batch_size-batch_leftover)//len(subdirectories)
    batch_take_from = batch_num*batch_size
    batch_take_until = batch_take_from+batch_size 

    print(batch_take_from,batch_take_until, batch_size)
    ground_truths = []
    #print(subdirectories)
    for subdir in subdirectories:#loop through directories of classes and get labels and vectorized versin of all imgs
        images_path = glob.glob(dir_path+'/'+subdir+'/*.jpg') + glob.glob(dir_path+'/'+subdir+'/*.png')
        np.random.shuffle(images_path)#shuffle dataset to avoid getting stuck
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
    return images_vectorized, ground_truths

def mnist_batches(train_imgs, train_labels, batch_size, batch_num):
    batch_take_from = batch_size*batch_num
    batch_take_until = batch_size*(batch_num+1)
    batch_imgs = train_imgs[batch_take_from:batch_take_until]
    # label = [0] * len(subdirectories)
    # label[subdirectories.index(subdir)] = 1
    # ground_truths.append([label]*(batch_size//2))#multiple label for each image in class
    batch_labels = np.array(train_labels[batch_take_from:batch_take_until])

    batch_imgs = np.array([img.reshape(-1)/255.0 for img in batch_imgs])

    #print(batch_labels)
    return batch_imgs, batch_labels


if __name__=="__main__":
    neuron_ops = neural_network_operations()
 
    testing_path = 'test'
    train_path = 'train'
    batch_size = 128



    weights, input_biases, num_hidden_layers, classes = neuron_ops.model_loader()
    (batch_train_imgs,batch_train_labels) , (batch_test, test_label) = mnist.load_data()

    #img = np.array([ImageOps.grayscale(Image.open('valid/kag_7786.png').resize((35,35)))])
    # ind = 16
    # img = batch_test[ind]
    # img = np.array([img.reshape(-1)/255.0])
    # ground_truth = to_categorical(test_label)[ind]
    # print(classes)
    # #print(img)
    # print(ground_truth)
    # #print(img[0])

    # all_outputs, output_layer_activations, weights, biases = neuron_ops.feed_forward(img[0], num_hidden_layers, len(classes), weights, input_biases)
    # print(["{:.5f}".format(val) for val in output_layer_activations])
    # print(output_layer_activations)

    # plt.imshow(batch_test[ind], cmap='gray')
    # plt.show()
    #neuron_ops.model_trainer(batch_size, 5)
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
