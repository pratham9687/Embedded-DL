#include <iostream>
#include <cmath>
#include "my_network.h"

#define ReLU 0
#define Sigmoid 1
#define TanAh 2

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

float * feed_forward(float * input){
    int weight_index = 0;
    int bias_index = 0;
    float * layer_output = nullptr;
    float * layer_input = nullptr;

    for(int i=0; i<num_layers; i++){

        int input_length;
        input_length = (i==0) ? input_resol : num_neurons_layers[i-1];
        
        float* current_input = (i==0) ? input : layer_output;
        float* current_output = new float[num_neurons_layers[i]];
        
        for (int j = 0; j < num_neurons_layers[i]; j++){
            float Z = 0;
            
            for(int k = 0; k < input_length; k++){
                Z +=current_input[k]*weights[weight_index];
                weight_index++;
            }

            Z += bias[bias_index];
            bias_index++;
            current_output[j] = activation_function(Z, ReLU);
        }
        if(i > 0) {
            delete[] layer_output;
        }

        layer_output = current_output;
    }
    return layer_output;
}

float input[] = {1.28776, -1.3376, 1.61069};
float * output;
int main(){
    for(int j=0; j<10; j++){
        //float input[] = {1.28776, -1.3376, 1.61069};
        output = feed_forward(input);
        for(int i=0; i<2; i++){
            std::cout << output[i] << " ";
        }
        delete[] output;
    }
    return 0;
}