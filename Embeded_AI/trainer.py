import pandas as pd
import Deep_learning
import matplotlib.pyplot as plt

df = pd.read_csv("house_data_balanced.csv")

X = df[["square_feet", "bedrooms", "house_age"]].values
y = df[["price","rent"]].values  # 1D output

X = (X - X.mean(axis=0)) / X.std(axis=0)  # Standardization
y = (y - y.mean(axis=0)) / y.std(axis=0)  # Standardize output too

# print(X, y)

my_network = Deep_learning.Neural_network()

params = [4, 3, 32, 64, 32, 2]
 # [num layers, input resolution, num neurons in L1, num neurons in L2, num neurons in L3, num neurons in L4]
my_network.generate_random_network(params)

# my_network.read_from_file_txt("my_network2")

costs = Deep_learning.back_prop(X, y, 0.001, my_network.layers, "Adam")

plt.plot(costs)
plt.show()

# my_network.update_network_file_txt("my_network2")
# my_network.update_network_file_ino()
# my_network.update_network_file_cpp()

# network_output = Deep_learning.forward_prop(my_network.layers, X)