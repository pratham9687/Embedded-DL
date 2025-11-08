import Deep_learning
from nnfs.datasets import sine_data, vertical_data, spiral_data
import numpy as np
import matplotlib.pyplot as plt

X, y = spiral_data(100,3)
Y = []
for i in y:
    data_line = [0,0,0]
    data_line[i] = 1
    Y.append(data_line)

Y = np.array(Y)

# Create a deeper network with more neurons

my_network = Deep_learning.Neural_network()
my_network.read_from_file_txt("my_network1")

# costs = Deep_learning.back_prop(X, Y, 0.1, my_network.layers)

#Plotting
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 3, 1)
# plt.plot(costs)
# plt.title('Training Cost Over Time')
# plt.xlabel('Epoch')
# plt.ylabel('Cost')
# plt.yscale('log')


output = Deep_learning.forward_prop(my_network.layers, X)
# print(my_network.layers[0].weights, my_network.layers[0].bias)
classifier = []
# my_network.update_network_file_txt("my_network1")

for o_p in output:
    classifier.append(np.argmax(o_p))
plt.subplot(1, 3, 2)
plt.scatter(X[:,0], X[:,1], c=classifier, cmap="brg")
plt.title('Spiral aproximation')
plt.subplot(1, 3, 3)
plt.scatter(X[:,0], X[:,1], c=y, cmap="brg")
plt.title('Spiral real')
plt.show()