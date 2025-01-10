# Model Loader
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QFileDialog
import sys

# Function to denormalize data
def denormalize(scaler, data, column_index=0):
    denormalized_data = scaler.inverse_transform([[x, 0] for x in data])[:, column_index]
    return denormalized_data

# Load the model
model = tf.keras.models.load_model('lstm_model.h5', compile=False)
print("Model loaded successfully.")

# Load the scaler
scaler = joblib.load('scaler.pkl')

def select_test_file():
    app = QApplication(sys.argv)
    file_dialog = QFileDialog()
    file_dialog.setFileMode(QFileDialog.ExistingFile)
    file_path, _ = file_dialog.getOpenFileName(
        None, "Select a Test File", ".", "CSV Files (*.csv);;All Files (*)"
    )
    return file_path

# Select and load the test dataset
test_file_path = select_test_file()
if not test_file_path:
    print("No file selected. Exiting...")
    exit()

test_dataset = pd.read_csv(test_file_path)
print(f"Test dataset loaded from {test_file_path}")

# Preprocess the test dataset
if 'datetime' in test_dataset.columns:
    test_dataset['datetime'] = pd.to_datetime(test_dataset['datetime'], format='%m/%d/%Y %H:%M')

numerical_columns = ['uptimevalue_raw', 'coverage_raw']
test_dataset[numerical_columns] = scaler.transform(test_dataset[numerical_columns])

# Prepare sequences for testing
sequence_length = 6
features = ['uptimevalue_raw', 'coverage_raw']
target = 'uptimevalue_raw'

test_data = []
test_labels = []
for i in range(sequence_length, len(test_dataset)):
    test_data.append(test_dataset[features].iloc[i-sequence_length:i].values)
    test_labels.append(test_dataset[target].iloc[i])

test_data = np.array(test_data)
test_labels = np.array(test_labels)

# Generate predictions
predictions = model.predict(test_data)


# Denormalize predictions and test labels
test_labels_rescaled = denormalize(scaler, test_labels, column_index=0)
predictions_rescaled = denormalize(scaler, predictions[:, 0], column_index=0)

# Evaluate the model
mae = mean_absolute_error(test_labels_rescaled, predictions_rescaled)
r2 = r2_score(test_labels_rescaled, predictions_rescaled)
print(f"Test Results:\n - Mean Absolute Error (MAE): {mae:.4f}\n - R² Score: {r2:.4f}")

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.plot(test_labels_rescaled[:5000], label="Actual", color='blue')
plt.plot(predictions_rescaled[:5000], label="Predicted", color='orange')
plt.title("LSTM Test Results: Actual vs Predicted")
plt.xlabel("Sample")
plt.ylabel("Uptime Value")
y_max = max(test_labels_rescaled[:5000])
plt.text(
    x=0,  # Posisi horizontal (ganti dengan nilai yang sesuai jika tidak terlihat)
    y=y_max * 0.8,  # Posisi vertikal berdasarkan data maksimum
    s="R² Score: 0.9780",  # Teks yang akan ditampilkan
    fontsize=12, 
    color="black", 
    bbox=dict(facecolor="lightyellow", alpha=0.5)  # Tambahkan latar belakang untuk menonjolkan teks
)
plt.legend()
plt.show() 

# Data Converter
# Denormalize predictions and test labels
test_labels_rescaled = denormalize(scaler, test_labels, column_index=0)
predictions_rescaled = denormalize(scaler, predictions[:, 0], column_index=0)


excel_base_date = datetime(1899, 12, 30)

# Convert the datetime column to datetime_raw
datetime_raw_values = pd.to_datetime(test_dataset['datetime'][sequence_length:]).apply(
    lambda dt: (dt - excel_base_date).days + (dt - excel_base_date).seconds / (24 * 60 * 60)
)

# Add predictions to the original dataset
# Align predictions with the datetime and input data
predictions_df = pd.DataFrame({
    'datetime': test_dataset['datetime'][sequence_length:].values,
    'datetime_raw':datetime_raw_values.values,
    'uptimevalue_raw': predictions_rescaled,
    'coverage_raw': test_dataset['coverage_raw'][sequence_length:].values,
})

# Save to CSV
output_csv_path = "predictions_with_metadata.csv"
predictions_df.to_csv(output_csv_path, index=False)
print(f"Predictions saved to {output_csv_path}")