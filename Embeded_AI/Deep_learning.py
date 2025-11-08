## Curently this tool has only 2 types of Optimiser
## 1) Adam 
## 2) LRS with momentum (Learning rate scheduling with momentum) 

import numpy as np
import matplotlib.pyplot as plt
import os
import platform


class Layer:
    def __init__(self, previous_layer_length, num_neurons):
        self.output = None
        self.input = None
        self.Z = None
        self.weights = []
        self.bias = []
        self.previous_layer_length = previous_layer_length
        self.num_neurons = num_neurons
        self.it_exists = False
        # Add momentum terms
        self.weight_momentum = None #np.zeros_like(self.weights)
        self.bias_momentum = None #np.zeros_like(self.bias)
        self.squared_momentum_weight = None
        self.squared_momentum_bias = None
        
    def initialize_layer(self):
        # Xavier/Glorot initialization for better gradient flow
        if not self.it_exists:
            self.weights = np.random.randn(self.num_neurons, self.previous_layer_length) * np.sqrt(2.0 / self.previous_layer_length)
            self.bias = np.zeros(self.num_neurons)  # Initialize biases to zero
            self.weight_momentum = np.zeros_like(self.weights)
            self.bias_momentum = np.zeros_like(self.bias)
            self.squared_momentum_weight = np.zeros_like(self.weights)
            self.squared_momentum_bias = np.zeros_like(self.bias)

            self.it_exists = True    

    def append_neuron(self, txt_data:str):
        # self.weights = []
        # self.bias = []
        
        if txt_data == "done":

            self.it_exists = True

            self.weights = np.array(self.weights)
            self.weight_momentum = np.zeros_like(self.weights)
            self.squared_momentum_weight = np.zeros_like(self.weights)

            self.bias = np.array(self.bias)
            self.bias_momentum = np.zeros_like(self.bias)
            self.squared_momentum_bias = np.zeros_like(self.bias)


        elif not self.it_exists:
            neueon_param = list(map(float, txt_data.split()))
            weights = list()
            for i in range(0,len(neueon_param)-1):
                weights.append(neueon_param[i])
            self.bias.append(neueon_param[len(neueon_param)-1])
            self.weights.append(weights)
            # print(self.weights)


    def tanh(self, value):
        return np.tanh(value)

    def ReLU(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def diff_tanh(self, value):
        return 1.0 - np.tanh(value)**2
    
    def diff_ReLU(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)


    def get_output(self, input):
        self.input = input
        self.Z = np.dot(input, self.weights.T) + self.bias
        self.output = self.ReLU(self.Z)  
        return self.output

def gradient_descent(layer, dC_wrt_da, inpt, optimiser:str, beta=0.9):
    da_wrt_dz = layer.diff_ReLU(layer.Z)
    dC_wrt_dz = dC_wrt_da * da_wrt_dz
    
    # Compute gradients
    dC_wrt_dw = np.mean([np.outer(dz, in_val) for in_val, dz in zip(inpt, dC_wrt_dz)], axis=0)
    dC_wrt_db = np.mean(dC_wrt_dz, axis=0)
        
    if optimiser == "LRS":
        # Update momentum
        layer.weight_momentum = beta * layer.weight_momentum + (1 - beta) * dC_wrt_dw
        layer.bias_momentum = beta * layer.bias_momentum + (1 - beta) * dC_wrt_db
        dC_wrt_da_prev_layer = np.dot(dC_wrt_dz, layer.weights)
        return layer.weight_momentum, layer.bias_momentum, dC_wrt_da_prev_layer
    elif optimiser == "Adam":
        # Calculating Squared momentum for RMS 
        layer.squared_momentum_weight = beta * layer.squared_momentum_weight + (1 - beta)*np.power(dC_wrt_dw,2)
        layer.squared_momentum_bias = beta * layer.squared_momentum_bias + (1 - beta)*np.power(dC_wrt_db, 2)
        # Calculating momentum 
        layer.weight_momentum = beta * layer.weight_momentum + (1 - beta) * dC_wrt_dw
        layer.bias_momentum = beta * layer.bias_momentum + (1 - beta) * dC_wrt_db
        # Calculating change in cost with respect to previous activation
        dC_wrt_da_prev_layer = np.dot(dC_wrt_dz, layer.weights)
        # Returning the values
        return layer.weight_momentum, layer.bias_momentum, dC_wrt_da_prev_layer, layer.squared_momentum_weight, layer.squared_momentum_bias, beta

def forward_prop(layers, input):
    for layer in layers:
        input = layer.get_output(input)
    return input

def back_prop(input_data, expected_output, initial_alpha, layers, optimiser = "LRS", epochs=1500):
    costs = []
    best_cost = float('inf')
    patience = 50
    patience_counter = 0
    min_alpha = 0.0001
    
    # Create wider network
    for epoch in range(epochs):

        # Forward pass
        network_output = forward_prop(layers, input_data)
        network_cost = np.mean(np.square(network_output - expected_output))
        costs.append(network_cost)
        
        # Early stopping check
        if network_cost < best_cost:
            best_cost = network_cost
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter > patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
        # Backward pass
        dc_wrt_da = 2 * (network_output - expected_output)
        for layer in reversed(layers):
            if optimiser == "LRS":
                # Implement learning rate decay with minimum value
                alpha = max(initial_alpha / (1 + epoch/1000), min_alpha)
                dc_wrt_dw, dc_wrt_db, dc_wrt_da = gradient_descent(layer, dc_wrt_da, layer.input, optimiser)
                layer.weights -= alpha * dc_wrt_dw
                layer.bias -= alpha * dc_wrt_db
            elif optimiser == "Adam":
                dc_wrt_dw, dc_wrt_db, dc_wrt_da, squared_momentum_weight, squared_momentum_bias, beta = gradient_descent(layer, dc_wrt_da, layer.input, optimiser)
                
                dc_wrt_dw_cap = dc_wrt_dw/(1-beta)
                squared_momentum_weight_cap = squared_momentum_weight/(1-beta)
                alpha = initial_alpha/(np.power(squared_momentum_weight_cap, 0.5) + 0.00000001)
                layer.weights -= alpha * dc_wrt_dw_cap
                
                dc_wrt_db_cap = dc_wrt_db/(1-beta)
                squared_momentum_bias_cap = squared_momentum_bias/(1-beta)
                alpha = initial_alpha/(np.power(squared_momentum_bias_cap, 0.5) + 0.00000001)
                layer.bias -= alpha * dc_wrt_db_cap
            
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Cost: {network_cost:.6f}, Learning Rate: {alpha}")
    
    return costs


class Neural_network:
    def __init__(self):
        self.layers = []
        self.num_layer = 0
        self.input_resolution = 0
        self.num_neurons = []
    
    def read_from_file_txt(self, network_name):
        # Opeaning the network file 
        with open(network_name+".txt", 'r') as network_file:
            parameters_txt = network_file.readline()
            param_list = parameters_txt.split()
            param_list = list(map(float, param_list))
            if param_list[0] == 1:
                # Getting the specifications of the network (headers of the network)
                self.num_layer = param_list[1]
                self.input_resolution = param_list[2]
                # print(param_list)
                for i in range(3,len(param_list)):
                    self.layers.append(Layer(param_list[i-1], param_list[i]))
                    self.num_neurons.append(param_list[i])
                    # Reading the parameters of the network
                    for j in range(0, int(param_list[i])):
                        neuron_data = network_file.readline()
                        # print(type(self.layers[i-3]))
                        self.layers[i-3].append_neuron(neuron_data)
                    self.layers[i-3].append_neuron("done")
                    # print(self.layers[i-3].weights)
    
    def update_network_file_txt(self, network_name):
        with open(network_name+".txt", 'w') as network_file:
            network_header = ""
            network_header = network_header + "1 "
            network_header = network_header + str(int(self.num_layer)) + " "
            network_header = network_header + str(int(self.input_resolution)) + " "
            for num_neurons in self.num_neurons:
                network_header = network_header + str(int(num_neurons)) + " "
            network_file.write(network_header + "\n")
            for i in range(int(self.num_layer)):
                for j in range(int(self.num_neurons[i])):
                    neuron_data = ""
                    for weight in self.layers[i].weights[j]:
                        neuron_data = neuron_data + str(weight) + " "
                    neuron_data = neuron_data + str(self.layers[i].bias[j])
                    network_file.write(neuron_data + "\n")

    def update_network_file_cpp(self):
        with open("my_network.h",'w') as network_file:
            start_header_guard = "#ifndef MNN \n#define MNN \n"
            end_header_guard = "#endif \n"
            num_layer_cmd = "int num_layers = "+str(int(self.num_layer))+"; \n"
            input_resol_info = "int input_resol = "+str(int(self.input_resolution))+"; \n"
            feed_forward_function = "float * feed_forward(float * input);\n"
            num_neuron_layer_info = "int num_neurons_layers[] = {"
            for i in range(len(self.num_neurons)):
                if i < len(self.num_neurons) - 1:
                    num_neuron_layer_info += str(int(self.num_neurons[i])) + ","
                else:
                    num_neuron_layer_info += str(int(self.num_neurons[i])) + "}; \n"
            c_weights = "const float weights[] = {"
            for i in range(len(self.layers)):
                for j in range(len(self.layers[i].weights)):
                    for k in range(len(self.layers[i].weights[j])):
                        if i < len(self.layers)-1 or j < len(self.layers[i].weights)-1 or k < len(self.layers[i].weights[j])-1:
                            c_weights += str(self.layers[i].weights[j][k])+","
                        else:
                            c_weights += str(self.layers[i].weights[j][k])+"}; \n"
            c_bias = "const float bias[] = {"
            for i in range(len(self.layers)):
                for j in range(len(self.layers[i].bias)):
                    if i < len(self.layers)-1 or j<len(self.layers[i].bias)-1:
                        c_bias += str(self.layers[i].bias[j])+","
                    else:
                        c_bias += str(self.layers[i].bias[j]) + "}; \n"
            file_content = start_header_guard + num_layer_cmd + input_resol_info + num_neuron_layer_info + c_weights + c_bias + feed_forward_function + end_header_guard
            network_file.write(file_content)

    def find_arduino_library_folder(self):
        system = platform.system()

        if system == "Windows":
            base_path = os.path.expanduser("~/Documents/Arduino/libraries/")
        elif system == "Linux":
            base_path = os.path.expanduser("~/Arduino/libraries/")
        elif system == "Darwin":  # macOS
            base_path = os.path.expanduser("~/Documents/Arduino/libraries/")
        else:
            raise OSError("Unsupported operating system")

        # Ensure the path exists
        if os.path.exists(base_path):
            return base_path.replace("\\", "/")
        else:
            print("Arduino library folder not found. Please specify manually.")
            return None


    def update_network_file_ino(self):
        libraries_path = self.find_arduino_library_folder()
        with open(libraries_path+"/FFNN_avr/FFNN_avr.h",'w') as network_file:
            start_header_guard = "#ifndef MNN \n#define MNN \n"
            header_files = "#include<Arduino.h> \n#include<avr/pgmspace.h> \n"
            end_header_guard = "#endif \n"
            num_layer_cmd = "const int num_layers = "+str(int(self.num_layer))+"; \n"
            input_resol_info = "const int input_resol = "+str(int(self.input_resolution))+"; \n"
            feed_forward_function = "float * feed_forward(float * input);\n"
            num_neuron_layer_info = "constexpr int num_neurons_layers[] = {"
            for i in range(len(self.num_neurons)):
                if i < len(self.num_neurons) - 1:
                    num_neuron_layer_info += str(int(self.num_neurons[i])) + ","
                else:
                    num_neuron_layer_info += str(int(self.num_neurons[i])) + "}; \n"
            c_weights = "const float weights[] PROGMEM = {"
            for i in range(len(self.layers)):
                for j in range(len(self.layers[i].weights)):
                    for k in range(len(self.layers[i].weights[j])):
                        if i < len(self.layers)-1 or j < len(self.layers[i].weights)-1 or k < len(self.layers[i].weights[j])-1:
                            c_weights += str(self.layers[i].weights[j][k])+","
                        else:
                            c_weights += str(self.layers[i].weights[j][k])+"}; \n"
            c_bias = "const float bias[] PROGMEM = {"
            for i in range(len(self.layers)):
                for j in range(len(self.layers[i].bias)):
                    if i < len(self.layers)-1 or j<len(self.layers[i].bias)-1:
                        c_bias += str(self.layers[i].bias[j])+","
                    else:
                        c_bias += str(self.layers[i].bias[j]) + "}; \n"
            file_content = start_header_guard + header_files + num_layer_cmd + input_resol_info + num_neuron_layer_info + c_weights + c_bias + feed_forward_function + end_header_guard
            network_file.write(file_content)


    def generate_random_network(self, parameters):
        self.num_layer = parameters[0]
        self.input_resolution = parameters[1]
        self.num_neurons = parameters[2:len(parameters)]
        for i in range(0,self.num_layer):
            if i==0:
                self.layers.append(Layer(self.input_resolution, self.num_neurons[i]))
            else:
                self.layers.append(Layer(self.num_neurons[i-1], self.num_neurons[i]))
            
            self.layers[i].initialize_layer()

if __name__ == "__main__":
    from nnfs.datasets import sine_data, vertical_data, spiral_data
    X, y = spiral_data(100,3)
    Y = []
    for i in y:
        data_line = [0,0,0]
        data_line[i] = 1
        Y.append(data_line)

    Y = np.array(Y)

    # Create a deeper network with more neurons

    my_network = Neural_network()
    params = [4, 2, 32, 64, 32, 3]
    my_network.generate_random_network(params)

    # Train the network
    costs = back_prop(X, Y, 0.001, my_network.layers, "Adam")

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(costs)
    plt.title('Training Cost Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.yscale('log') 

    plt.subplot(1, 3, 2)
    output = forward_prop(my_network.layers, X)
    classifier = []
    my_network.update_network_file_txt("my_network1")

    for o_p in output:
        classifier.append(np.argmax(o_p))
    plt.scatter(X[:,0], X[:,1], c=classifier, cmap="brg")
    plt.title('Spiral aproximation')
    plt.subplot(1, 3, 3)
    plt.scatter(X[:,0], X[:,1], c=y, cmap="brg")
    plt.title('Spiral real')
    plt.show()