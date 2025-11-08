import pandas as pd
import csv

df = pd.read_csv("house_data_balanced.csv")

# Extract features and target values
# X = df[["square_feet", "bedrooms", "house_age"]].values
# y = df[["price"]].values  # 1D output
# Y = y
# Compute mean and standard deviation before standardization
X = df[["square_feet","bedrooms","house_age","price","rent"]].values
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)

# y_mean = y.mean(axis=0)
# y_std = y.std(axis=0)

# Standardization
x = (X - X_mean) / X_std  
# y = (y - y_mean) / y_std  

with open("normal_data.csv", "w", newline="") as f:
    x = list(x)
    writer = csv.writer(f)
    writer.writerows(x)
