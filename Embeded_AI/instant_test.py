import numpy as np

my_arr = np.array([[1,2],[3,4]])
my_arr2 = np.array([2,5])
my_arr3 = np.array([-1,1,2])
print(my_arr)
print(my_arr.T)
print(my_arr2*my_arr.T)
print(np.sum(my_arr2*my_arr.T,axis=1))
# print((my_arr3>0))
# print(my_arr2.T.dot(my_arr3))
string = "hello"
print(string+" world")

my_str = "10, 20"
my_list = [float(resp) for resp in my_str.split(",")]

print(my_list)