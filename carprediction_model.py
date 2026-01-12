# Car Price Prediction using Random Forest Regressor (with ZIP extraction)

import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Path to your ZIP file
zip_path = r"C:\Users\k3019\Downloads\archive (1).zip"

# Extract ZIP contents to a folder
extract_folder = r"C:\Users\k3019\Downloads\car_data_extracted"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

# Show extracted files
print("Extracted files:", os.listdir(extract_folder))

# Load dataset (update filename if different inside ZIP)
csv_file = os.path.join(extract_folder, "car data.csv")
try:
    df = pd.read_csv(csv_file)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: CSV file not found inside ZIP. Check extracted files list above.")
    exit()

# Display first rows
print(df.head())

# Drop Car_Name column (not useful for prediction)
if "Car_Name" in df.columns:
    df = df.drop("Car_Name", axis=1)

# Convert categorical columns to numerical
df = pd.get_dummies(df, drop_first=True)

# Features and target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("R2 Score:", r2)

# Visualization: Actual vs Predicted Car Prices
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, alpha=0.7, color="blue", label="Predicted")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    label="Perfect Prediction"
)
plt.xlabel("Actual Car Price")
plt.ylabel("Predicted Car Price")
plt.title("Car Price Prediction using Random Forest")
plt.legend()
plt.grid(True)
plt.show()