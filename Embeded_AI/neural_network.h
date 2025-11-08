#ifndef NN_H
#define NN_H

#include <sstream>
#include <string>
#include <fstream>  // For file handling
#include <cmath>

#define ReLU 0
#define Sigmoid 1
#define TanAh 2

struct Layer_data{
    float ** weights;
    float * bias;
    int num_weights;
    int num_neurons;
};

Layer_data * read_layers(std::string network_name);
float** read_dataset(int data_length, int data_width, std::string file_name); // Temp function
float activation_function(float Z, int activation_type, float alpha);
float * feed_forward(float* input, Layer_data * layers, int activation_type);
void delete_network(Layer_data * layers);

#endif