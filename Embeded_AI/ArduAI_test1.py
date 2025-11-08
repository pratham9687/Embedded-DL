import pandas as pd
import serial
import time
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("house_data_balanced.csv")

# print(df)

X = df[["square_feet", "bedrooms", "house_age"]].values
y = df[["price","rent"]].values  # 2D output

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)

y_mean = y.mean(axis=0)
y_std = y.std(axis=0)

x = (X - X_mean) / X_std  # Standardization
# y = (y - y_mean) / y_std

ser = serial.Serial('COM4', 9600, timeout=1)

time.sleep(5)

Y = []
i=0
for x_1d in x:
    data = f"{x_1d[0]},{x_1d[1]},{x_1d[2]}\n"
    print("Sending data : ", data)
    ser.write(data.encode())

    while not ser.in_waiting:
        pass
    if ser.in_waiting > 0:
        response = ser.readline().decode('utf-8').strip()
        outputs = [float(resp) for resp in response.split(",")]
        Y.append(outputs)
        # print(outputs)
        # print(y[i])
        i+=1
        
    
    time.sleep(0.015)

original_values = y
network_output = (np.array(Y)*y_std)+y_mean

ser.close()

plt.figure(figsize=(8, 5))
plt.plot(original_values, 'bo-', label="Original Values")
plt.plot(network_output, 'r*-', label="Predicted Values")
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Original vs Predicted Values")
plt.legend()
plt.grid(True)
plt.show()