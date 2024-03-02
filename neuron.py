import numpy as np
from PIL import Image, ImageOps
# from numba import jit, cuda
# from numba.experimental import jitclass
from timeit import default_timer as timer  
import os
#piss off tensorflow warnings, thanks very mucho
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

import json
import concurrent.futures as cf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

class neural_network_operations():

    def relu(self, activatables):#the activation function
        return np.maximum(0, activatables)

    def weighted_sum_output(self, inputs, weights, bias):#single neuron output, sum of inputs*weights -> scalar
        if inputs.ndim == 1:#inputs are of type vector
            if weights.ndim == 1:#assuming one neuron per layer
                output = np.dot(weights, inputs)
                output += bias
            else:
                raise ValueError("Weights must be of type vector")
        else:
            raise ValueError("inputs must be of type vector")
        return output

    def weighted_sum_layer(self, inputs, weights, biases):#for multiple neurons in layer (could be hidden layer or output layuer)
        if weights.ndim == 2:#assuming multiple neurons per layer
            outputs = np.matmul(weights,inputs)+biases
        else:
            raise ValueError("Weights must be 2D matrix")
        return outputs.reshape(len(biases), 1)

    
    def sigmoid(self, output_layer_activations):
        sigmoid = np.array([])
        print(np.exp(-output_layer_activations[0]))
        for activation in output_layer_activations:
            probability =  0.0#
            #sum_activation_probs = 0.0            
            probability = np.exp(-activation)/(1+np.exp(-activation))**2
            sigmoid = np.append(sigmoid, probability)
        return sigmoid

    def feed_forward(self, inputs, num_hidden_layers:int, output_layer_size:int, weights=None, biases=None, weight_size=64):
        all_outputs = []
        all_activations = [inputs]
        if weights == None:
            first_hidden_layer_weights = np.random.uniform(0.0, np.sqrt(2/len(inputs)), size=(weight_size, len(inputs)))#first hidden takes in the input features
            weights = [np.random.uniform(0.0, np.sqrt(2/len(inputs)), size=(weight_size, weight_size)) for _ in range(num_hidden_layers-1)] #ignore first hidden layer weights
            weights.insert(0, first_hidden_layer_weights)
            output_layer_weights = np.random.uniform(0.0, np.sqrt(2/len(inputs)), size=(output_layer_size, weight_size))
            weights.append(output_layer_weights)
            biases = [np.array([np.random.uniform(0.0, np.sqrt(2/len(inputs)), size=(weight_size))]).T for _ in range(num_hidden_layers)]
            output_layer_biases = np.array([np.random.uniform(0.0, np.sqrt(2/len(inputs)), size=(output_layer_size))]).T
            biases.append(output_layer_biases)
        ind = 0

        #pass layer activations through all hidden layers and calculates activations
        while num_hidden_layers > 0:#0 = output layer
            hidden_layer_outputs = self.weighted_sum_layer(inputs, weights[ind], biases[ind])
            all_outputs.append(hidden_layer_outputs) #size = 1xnum_hidden_layers
            hidden_layer_activations = self.relu(hidden_layer_outputs)
            all_activations.append(hidden_layer_activations)
            inputs = hidden_layer_activations
            num_hidden_layers -= 1
            ind += 1  
        output_layer_outputs = self.weighted_sum_layer(inputs, weights[ind], biases[ind])
        all_outputs.append(output_layer_outputs)
        output_layer_activations = self.softmax(output_layer_outputs)
        all_activations.append(output_layer_activations)
        return all_outputs, all_activations, weights, biases

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
            loss = -np.sum((expected_layer*np.log(probabilities+epsilon)))
        return loss
    
    def cross_entropy_loss_backpropagation(self, output_layer, expected_layer):
        probabilities = output_layer
        epsilon = 1e-10
        loss = -(expected_layer/(probabilities+epsilon))
        return loss
    
    def relu_backpropagation(self, activatables) -> 0 | 1:#derivative of relu
        return (activatables > 0).astype(np.int32)
    
    def softmax(self, output_layer_activations):
        e_output = np.exp(output_layer_activations - np.max(output_layer_activations))
        return e_output / e_output.sum(axis=0)

    #redundant
    # def softmax_backpropagation(self, softmax):
    #         # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
    #     s = softmax.reshape(-1, 1)
    #     print(s.shape, np.diagflat(s).shape)

    #     return np.diagflat(s) - np.dot(s, s.T)
        #return self.softmax(activatables)*(1-self.softmax(activatables))
    
    def cross_entropy_softmax_derivative(self, output_probs, expected):
        return output_probs-expected
        
    def sigmoid_backpropagation(self, activatables):
        return self.sigmoid(activatables)*(1-self.sigmoid(activatables))
    
    def hadamard_product(self, cost_gradient, activations_backpropagation):
        if isinstance(cost_gradient, (int, float)):#if only one gradient value
            return cost_gradient*activations_backpropagation
        else:
            hadamard_res = np.multiply(cost_gradient, activations_backpropagation)
        return hadamard_res.reshape(len(activations_backpropagation), 1)

    def output_layer_errors(self, output_layer_outputs: np.array, output_layer_activations: np.array, expected_layer: np.array) -> np.array:#gets the last layer's errors
        errors = self.cross_entropy_softmax_derivative(output_layer_activations, expected_layer) #relu derivative
        return errors
    
    def layer_errors(self, previous_layer_weights, previous_layer_errors, current_layer_outputs):        
        weighted_sum_errors = np.matmul(previous_layer_weights.T, previous_layer_errors)
        layer_errors = self.hadamard_product(weighted_sum_errors, self.relu_backpropagation(current_layer_outputs))
        return layer_errors
    
    def gradient_descent(self, input_weights, input_biases, gradient_weights,  gradient_biases, learning_rate):
        for ind, (weights, biases) in enumerate(zip(gradient_weights, gradient_biases)):#each batch element's set of weights and biases
            input_weights[ind] -= learning_rate*weights
            input_biases[ind] -= learning_rate*biases
        return input_weights, input_biases

    def backpropagation(self, all_outputs, all_activations, expected_layer, all_weights):
        num_layers = len(all_weights)-1
        output_layer_errors = self.output_layer_errors(all_outputs[-1], all_activations[-1], expected_layer)
        previous_layer_errors = self.layer_errors(all_weights[-1], output_layer_errors,  all_outputs[-2])
        all_cost_wrt_biases = [output_layer_errors, previous_layer_errors]

        # Vectorized computation of all_cost_wrt_biases
        all_cost_wrt_biases += [self.layer_errors(all_weights[layer+1], previous_layer_errors,  all_outputs[layer]) for layer in range(num_layers-2, -1, -1)]

        # Vectorized computation of all_cost_wrt_weights
        all_activations.pop(-1)
        all_cost_wrt_weights = [np.matmul(layer_errors, prev_layer_activations.T) for layer_errors, prev_layer_activations in zip(all_cost_wrt_biases, all_activations[::-1])]
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
                np.add(mean_weights[ind], weights, out=mean_weights[ind])#mean_weights[ind] += weights
                np.add(mean_biases[ind], biases, out=mean_biases[ind])#mean_biases[ind] += biases

        #after summation divide to get mean
        for ind in range(len(mean_biases)):
            np.divide(mean_weights[ind], len(batch_weights), out=mean_weights[ind])
            np.divide(mean_biases[ind], len(batch_biases), out=mean_biases[ind])
            #mean_weights[ind] /= len(batch_weights)
            #mean_biases[ind] /= len(batch_biases)
        return mean_weights, mean_biases
    
    def batch_trainer(self, batch_train_imgs, batch_train_labels, weights, biases, num_classes, num_hidden_layers, learning_rate, grayscale=True):
        batch_weights = []
        batch_biases = []
        average_loss = 0.0
        for train_img, batch_train_label in zip(batch_train_imgs, batch_train_labels):#for each train and test img, get new weights and biases
            if weights and biases:
                all_outputs, all_activations, weights, biases = self.feed_forward(train_img, num_hidden_layers, num_classes, weights, biases)
            else:
                all_outputs, all_activations, weights, biases = self.feed_forward(train_img, num_hidden_layers, num_classes)
            average_loss += self.cross_entropy_loss(all_activations[-1], batch_train_label)
            gradient_weights, gradient_biases = self.backpropagation(all_outputs, all_activations, batch_train_label, weights)
            train_example_weights, train_example_biases = self.gradient_descent(weights, biases, gradient_weights, gradient_biases,  learning_rate)
            batch_weights.append(train_example_weights)
            batch_biases.append(train_example_biases)
        weights, biases = self.batch_averages(batch_weights, batch_biases)#calculate average values for all weights and biases
        average_loss /= len(batch_train_imgs)
        return weights, biases, average_loss

    def get_num_batches(self, batch_train_imgs, batch_size):
        num_batches = 0
        while len(batch_train_imgs) > (batch_size*num_batches):#batch traun_imgs is a vector consisting of images
                num_batches += 1
        return num_batches

    def model_trainer(self, train_imgs, train_labels, batch_size, epochs=1, learning_rate=5e-3): 
        num_classes = len(train_labels[0])
        print(num_classes)
        num_hidden_layers = 1
        final_weights, final_biases = [], []
        total_avg_loss = 0.0
        print(num_classes, learning_rate)
        for epoch in range(1, epochs+1):#start at 1
            average_epoch_loss = np.array([])
            epoch_weights, epoch_biases = [], []

            print(f"Epoch/Epochs: {epoch}/{epochs}")
            epoch_train_imgs, epoch_train_labels, num_batches = self.batch_processor(train_imgs, train_labels, batch_size)
            epoch_results = self.execute_batches_in_parallel([epoch_train_imgs, epoch_train_labels, final_weights, final_biases, num_classes, num_hidden_layers, learning_rate])
            
            for batch_results in epoch_results:
                epoch_weights.append(batch_results[0])
                epoch_biases.append(batch_results[1])
                average_epoch_loss = np.append(average_epoch_loss, batch_results[2])
            final_weights, final_biases = self.batch_averages(epoch_weights, epoch_biases)
            
            average_epoch_loss = np.sum(average_epoch_loss)/num_batches
            total_avg_loss += average_epoch_loss
            print(f"Epoch loss: {average_epoch_loss}, Average Epoch Loss: {total_avg_loss/epoch}")
            if average_epoch_loss < total_avg_loss/epochs or epoch == 1:
                data_names = ['weights', 'biases', 'hidden_layers', 'classes']
                self.model_saver(data_names, final_weights, final_biases, num_hidden_layers)
        return final_weights, final_biases, num_hidden_layers
    
    def execute_batches_in_parallel(self, data):
        train_imgs_batches, train_labels_batches, weights, biases, num_classes, num_hidden_layers, learning_rate = data
        with cf.ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.batch_trainer, batch_train_imgs, batch_train_labels, weights, biases, num_classes, num_hidden_layers, learning_rate) for (batch_train_imgs, batch_train_labels) in zip(train_imgs_batches, train_labels_batches)]
            cf.wait(futures)
        results = [future.result() for future in cf.as_completed(futures)]

        return results

    def model_saver(self, data_names:list, *args, model_file_path="model_checkpoint.json"):
        subdirectories = [d for d in os.listdir('train') if os.path.isdir(os.path.join('train', d))]
        #label_mapping = {label: index for index, label in enumerate(subdirectories)}
        
        data_to_save = {}
        for name,arg in zip(data_names, args):
            if name != 'hidden_layers':
                arg = [arr.tolist() for arr in arg]
            data_to_save[name] = arg

        data_to_save['classes'] = subdirectories

        #print(data_to_save)
        with open(model_file_path, 'w') as json_file:
            json.dump(data_to_save, json_file)

    def model_loader(self, model_file_path = "model_checkpoint.json"):
                
        with open(model_file_path, 'r') as json_file:
            data = json.load(json_file)
        weights = [np.array(saved_weights) for saved_weights in data["weights"]]
        biases = [np.array(saved_biases) for saved_biases in data["biases"]]
        num_hidden_layers = data["hidden_layers"]
        classes = data["classes"]

        return weights, biases, num_hidden_layers, classes   
    
    def model_tester(self, test_imgs, test_labels, weights, biases, num_hidden_layers, num_classes):
        #self, inputs, num_hidden_layers:int, output_layer_size:int, weights=None, biases=None, weight_size=64
        num_correct = 0
        batch_test_imgs, batch_test_labels, _ = self.batch_processor(test_imgs, test_labels, batch_size)

        for test_img, test_label in zip(batch_test_imgs[0], batch_test_labels[0]):
            _, all_activations, _, _ = self.feed_forward(test_img, num_hidden_layers, num_classes, weights, biases)
            if np.argmax(all_activations[-1]) == np.argmax(test_label):
                num_correct += 1
        accuracy = num_correct/len(batch_test_imgs[0])
        print(f"Testing accuracy:  {accuracy} \nCorrect/Total: {num_correct}/{len(batch_test_imgs[0])}")
        return accuracy

    #handles shuffling and batch creation
    def batch_processor(self, imgs, labels, batch_size):#only works with cupy arrays
        num_batches = self.get_num_batches(imgs, batch_size)
        print("Num Batches: ", num_batches)
        indices = np.arange(imgs.shape[0])
        np.random.shuffle(indices)
        
        imgs_shuffled = imgs[indices]
        labels_shuffled = labels[indices]
        epoch_imgs, epoch_labels = [], []
        for batch_num in range(num_batches):
            batch_train_imgs,batch_train_labels = mnist_batches(imgs_shuffled, labels_shuffled, batch_size, batch_num)
            epoch_imgs.append(batch_train_imgs)
            epoch_labels.append(batch_train_labels)
        print(epoch_imgs[0].shape)
        return epoch_imgs, epoch_labels, num_batches
    

#handles loading images from subdirectories
def images_loader(dir_path, img_resize=35, num_imgs=0,grayscale=False, load_from_file=False, save_to_file=True):#loads a batch of images to be used
    try:
        if not load_from_file:
            raise Exception("Number of imgs changed, defaulting to loader")
        image_list = np.load(f'image_list_{dir_path}.npy')
        ground_truths = np.load(f'ground_truths_{dir_path}.npy')
        print("Found pre-saved images")
    except Exception as e:
        print(e)
        subdirectories = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
        num_classes= len(subdirectories)
        ground_truths = np.empty((0, num_classes))
        image_list = np.empty((0, img_resize, img_resize)) if grayscale else np.empty((0,img_resize,img_resize,3))
        print(subdirectories)
        for subdir in subdirectories:#loop through directories of classes and get labels and vectorized versin of all imgs
            imgs_path = os.path.join(dir_path, subdir)
            for (cnt, filename) in enumerate(os.listdir(imgs_path)):
                if num_imgs == 0 or cnt != num_imgs//num_classes:
                    if filename.endswith(".jpg") or filename.endswith(".png"):
                        img_path = os.path.join(imgs_path, filename)
                        if grayscale:
                            img = ImageOps.grayscale(Image.open(img_path).resize((img_resize, img_resize)))
                        else:
                            img = Image.open(img_path).resize((img_resize,img_resize))
                        img_array = np.asarray(img)
                        #code for removing non-RGB files
                        """if img_array.ndim != 3 or img_array.shape[2] != 3:
                            print(img_array.shape, img_path)
                            os.remove(img_path)
                            continue """
                        image_list = np.concatenate((image_list, img_array[np.newaxis, :, :]), axis=0)
                        label = np.zeros( (1, num_classes) )# * len(subdirectories)
                        label[0][subdirectories.index(subdir)] = 1
                        ground_truths = np.concatenate((ground_truths, label), axis=0)#multiple label for each image in class
                else:
                    break
        if save_to_file:
            np.save(f'image_list_{dir_path}.npy', image_list)
            np.save(f'ground_truths_{dir_path}.npy', ground_truths)
    print(len(ground_truths), len(image_list))
    return image_list, ground_truths

def mnist_batches(train_imgs, train_labels, batch_size, batch_num):
    #print(train_imgs[0], '-------------------',train_imgs_shuffled[0])

    batch_take_from = batch_size*batch_num
    batch_take_until = batch_size*(batch_num+1)
    batch_imgs = train_imgs[batch_take_from:batch_take_until]

    #reshape to form nx1
    batch_labels = np.array([train_label.reshape(len(train_label),1) for train_label in train_labels[batch_take_from:batch_take_until]])
    batch_imgs = np.array([(img.reshape(-1,1)/255.0) for img in batch_imgs])
    return batch_imgs, batch_labels


if __name__=="__main__":
    neuron_ops = neural_network_operations()
 
    test_path = 'test'
    train_path = 'train'
    batch_size = 128
    
    train_imgs, train_labels = images_loader(train_path, num_imgs=60000, grayscale=False, load_from_file=True, save_to_file=False)
    neuron_ops.model_trainer(train_imgs[:50000], train_labels[:50000], batch_size, 12)
    test_imgs, test_labels = images_loader(test_path, num_imgs=9, grayscale=False, load_from_file=False, save_to_file=False)

    #MNIST handwritten digits dataset
    # (train_imgs,train_labels) , (test_imgs, test_labels) = mnist.load_data()
    # train_labels, test_labels = to_categorical(train_labels), to_categorical(test_labels)#for softmax classification
    # train_imgs, train_labels, test_imgs, test_labels = np.asarray(train_imgs), np.asarray(train_labels), np.asarray(test_imgs), np.asarray(test_labels)

    model = "model_checkpoint.json"
    weights, biases, num_hidden_layers, classes = neuron_ops.model_loader(model)
    
    test_acc = neuron_ops.model_tester(test_imgs, test_labels, weights, biases, num_hidden_layers, 3)

    #visualization
    ind = 1
    reshape_to = (35,35,3)
    batch_test_imgs, batch_test_labels, _ = neuron_ops.batch_processor(test_imgs, test_labels, batch_size)
    test_img = batch_test_imgs[0][ind]
    _, all_activations, _, _ = neuron_ops.feed_forward(test_img, num_hidden_layers, 3, weights, biases)
    print(type(all_activations[-1]), type(classes),  all_activations[-1].ravel().ndim)
    preds_to_list = all_activations[-1].ravel().tolist()
    preds_to_list = [round(pred, 4) for pred in preds_to_list]
    print(f"Predicted values:  {dict(zip(classes, preds_to_list))} \nTrue values: {dict(zip(classes, batch_test_labels[0][ind].reshape(-1).ravel().tolist()))}")
    plt.imshow(np.asnumpy(test_img.reshape(reshape_to)), cmap='gray')
    plt.show()