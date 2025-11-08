import pandas as pd
import Deep_learning
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("house_data_balanced.csv")

# Extract features and target values
X = df[["square_feet", "bedrooms", "house_age"]].values
y = df[["price","rent"]].values  # 2D output
Y = y
# Compute mean and standard deviation before standardization
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)

y_mean = y.mean(axis=0)
y_std = y.std(axis=0)

# Standardization
X = (X - X_mean) / X_std  
y = (y - y_mean) / y_std  

# Initialize and load neural network
my_network = Deep_learning.Neural_network()
my_network.read_from_file_txt("my_network2")
my_network.update_network_file_ino()
# Forward propagation
network_output = Deep_learning.forward_prop(my_network.layers, X)

print(network_output)

# Function to denormalize output
def denormalize(y_normalized, y_mean, y_std):
    return y_normalized * y_std + y_mean

# Convert predicted values back to original price and rent
predicted_price = denormalize(network_output, y_mean, y_std)

# Print original predicted prices and rents
# print("Predicted Prices and Rents(Original Scale), Original Prices and Rents(Original Scale):")
# print(predicted_price, Y)

original_values = Y
predicted_values = predicted_price

plt.figure(figsize=(8, 5))
plt.plot(original_values, 'bo-', label="Original Values")
plt.plot(predicted_values, 'r*-', label="Predicted Values")
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Original vs Predicted Values")
plt.legend()
plt.grid(True)
plt.show()
