#include<iostream>
#include "neural_network.h"

int main(){
    Layer_data * my_neural_network;
    my_neural_network = read_layers("my_network2");
    //std::cout << my_neural_network[0].weights[0][1] << "\n";
    float** my_dataset;
    my_dataset = read_dataset(100, 5, "normal_data.csv");
    //std::cout << my_dataset[0][0] << "\n";
    float network_input[3];
    for(int i=0; i<3; i++){
        network_input[i] = my_dataset[0][i];
        std::cout << network_input[i] << " ";
    }
    std::cout << "\n";
    float * network_output = feed_forward(network_input, my_neural_network, ReLU);
    for(int i=0; i<2; i++) std::cout << network_output[i] << " ";
    delete[] network_output;
    delete_network(my_neural_network);
    return 0;
}