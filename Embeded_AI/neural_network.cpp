#include "neural_network.h"
#include<iostream>

// #define ReLU 0
// #define Sigmoid 1
// #define TanAh 2

int num_layers;
int input_resolution;


Layer_data * read_layers(std::string network_name){
    // Open the file containing the neural network
    Layer_data* layers;
    network_name = network_name + ".txt";
    std::ifstream network_file(network_name);
    // Read headers(parameters or specifications) of the neural network
    std::string line_1;
    std::getline(network_file, line_1);
    std::istringstream params(line_1);
    int status;
    params >> status;
    if(status){
        params >> num_layers;
        params >> input_resolution;
        layers = new Layer_data[num_layers];
        for (int i = 0; i < num_layers; i++){
            params >> layers[i].num_neurons;   
        }
    }
    // Read the Layer data (weights and biases) of the neural network 
    
    for(int i=0; i < num_layers; i++){
        layers[i].weights = new float*[layers[i].num_neurons];
        layers[i].bias = new float[layers[i].num_neurons];
        std::string neuron_data;           
        if(i==0) layers[i].num_weights = input_resolution;
        else layers[i].num_weights = layers[i-1].num_neurons;
        
        for(int j=0; j<layers[i].num_neurons; j++){
            layers[i].weights[j] = new float[layers[i].num_weights];
            std::getline(network_file, neuron_data);
            std::istringstream neuron_data_buffer(neuron_data);
            for(int k=0; k<layers[i].num_weights; k++){
                neuron_data_buffer >> layers[i].weights[j][k];
            }
            neuron_data_buffer >> layers[i].bias[j];        
        }
    }
    return layers;
}

float** read_dataset(int data_length, int data_width, std::string file_name) {
    float** data_set = new float*[data_length];
    std::ifstream data_file(file_name);
    
    if (!data_file.is_open()) {
        std::cerr << "Error: Could not open file " << file_name << std::endl;
        return nullptr;
    }
    
    for (int i = 0; i < data_length; i++) {
        data_set[i] = new float[data_width];
        std::string data_line;
        std::getline(data_file, data_line);
        
        // Replace commas with spaces for easier parsing
        for (size_t j = 0; j < data_line.length(); j++) {
            if (data_line[j] == ',') {
                data_line[j] = ' ';
            }
        }
        
        std::istringstream data_line_buffer(data_line);
        for (int j = 0; j < data_width; j++) {
            if (!(data_line_buffer >> data_set[i][j])) {
                std::cerr << "Error reading value at row " << i << ", column " << j << std::endl;
                // Set to default value
                data_set[i][j] = 0.0f;
            }
        }
    }
    
    data_file.close();
    return data_set;
}

float activation_function(float Z, int activation_type, float alpha = 0.01){
    if(activation_type == ReLU){
        return (Z > 0) ? Z : alpha * Z; 
    } else if(activation_type == Sigmoid){
        return 1.0f / (1.0f + exp(-Z));
    } else if(activation_type == TanAh){
        return tanh(Z);
    }
    return Z;  // Default fallback
}

float * feed_forward(float* input, Layer_data * layers, int activation_type) {  
    float * layer_output = nullptr;
    float * layer_input = nullptr;
    
    for(int i=0; i<num_layers; i++) {
        int input_size = (i==0) ? input_resolution : layers[i-1].num_neurons;
        int output_size = layers[i].num_neurons;
        
        // Create output array for this layer
        float* current_output = new float[output_size];
        
        // Use the appropriate input based on which layer we're on
        float* current_input = (i==0) ? input : layer_output;
        
        // Perform matrix multiplication, add bias and apply activation function
        for(int j=0; j<output_size; j++) {
            float Z = 0.0;
            for(int k=0; k<input_size; k++) {
                Z += current_input[k] * layers[i].weights[j][k];
            }
            Z += layers[i].bias[j];
            
            // Apply activation function
            if(activation_type == ReLU) {
                current_output[j] = (Z > 0) ? Z : 0.01 * Z;  // ReLU with leaky term
            } else if(activation_type == Sigmoid) {
                current_output[j] = 1.0 / (1.0 + exp(-Z));
            } else if(activation_type == TanAh) {
                current_output[j] = tanh(Z);
            }
        }
        
        // Clean up previous layer's output before moving to next layer
        if(i > 0) {
            delete[] layer_output;
        }
        
        // Save this layer's output for the next iteration
        layer_output = current_output;
    }
    
    return layer_output;
}

void delete_network(Layer_data * layers){
    for(int i=0; i<num_layers; i++){
        delete[] layers[i].bias;
        for(int j=0; j<layers[i].num_neurons; j++){
            delete[] layers[i].weights[j];
        } 
    }
    delete[] layers;
}