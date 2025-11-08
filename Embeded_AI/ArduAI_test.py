import pandas as pd
import matplotlib.pyplot as plt
import serial
import time
import numpy as np

df = pd.read_csv("house_data_balanced.csv")

X = df[["square_feet", "bedrooms", "house_age"]].values
y = df[["price","rent"]].values  # 1D output

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)

y_mean = y.mean()
y_std = y.std()

x = (X - X_mean) / X_std  # Standardization
y = (y - y_mean) / y_std

ser = serial.Serial('COM4', 9600, timeout=1)

time.sleep(5)

Y = []

for x_1d in x[0:5]:
    data = ""
    for i in range(len(x_1d)):
        if i<len(x_1d)-1:
            data += f"{x_1d[i]},"
        else:
            data += f"{x_1d[i]}\n"
        ser.write(data.encode())
    
    print("Sending data "+ data)
    
    try:
        while ser.in_waiting == 0:
            pass
        if ser.in_waiting > 0:
            response = ser.readline().decode('utf-8').strip()
            outputs = [float(resp) for resp in response.split(",")]
            if len(outputs) == 2:
                print(outputs)
                Y.append(outputs)

    except UnicodeDecodeError:
        print("Warning: Received data couldn't be decoded as text")
    except ValueError:
        print(f"Warning: Couldn't convert value to float in: {response}")
    
    time.sleep(0.1)

original_values = y[0:5]
network_output = np.array(Y)

# def denormalize(y_normalized, y_mean, y_std):
#     return y_normalized * y_std + y_mean

# # Convert predicted values back to original price and rent
# predicted_price = denormalize(network_output, y_mean, y_std)

# original_values = y
# predicted_values = Y

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