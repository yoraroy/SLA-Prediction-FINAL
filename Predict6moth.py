# Import necessary libraries
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

# Function to predict 6 months into the future with 5-minute intervals
def predict_six_months(model, scaler, dataset, start_date, sequence_length=6, steps=51840):
    """
    Predicts 6 months into the future with 5-minute intervals (51840 steps).
    """
    # Prepare the initial sequence from the dataset
    initial_sequence = dataset[-sequence_length:][['uptimevalue_raw', 'coverage_raw']].values
    initial_sequence = initial_sequence.reshape(1, sequence_length, -1)

    predictions = []
    current_sequence = initial_sequence.copy()

    for _ in range(steps):
        # Predict the next value
        next_pred = model.predict(current_sequence)[0, 0]
        predictions.append(next_pred)

        # Update the sequence: drop the oldest and add the new prediction
        next_sequence = np.append(current_sequence[:, 1:, :], [[[next_pred, 0]]], axis=1)
        current_sequence = next_sequence

    # Denormalize predictions
    predictions_rescaled = denormalize(scaler, predictions, column_index=0)

    # Generate corresponding dates for 5-minute intervals
    future_dates = [start_date + timedelta(minutes=5 * i) for i in range(steps)]

    # Prepare the results as a DataFrame
    predictions_df = pd.DataFrame({
        'datetime': future_dates,
        'uptimevalue_raw': predictions_rescaled
    })

    return predictions_df

# Load the model
model = tf.keras.models.load_model('./model/lstm_model.h5', compile=False)
print("Model loaded successfully.")

# Load the scaler
scaler = joblib.load('./model/scaler.pkl')

# Function to select test file
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

# Dynamically set start_date to the last date in the dataset or a custom date
use_last_date = True  # Set to False if you want to use a custom date
if use_last_date:
    start_date = pd.to_datetime(test_dataset['datetime'].iloc[-1])
else:
    start_date = pd.to_datetime("2025-01-01 00:00:00")  # Custom start date
print(f"Using start date: {start_date}")

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

# Generate predictions for the test dataset
predictions = model.predict(test_data)

# Denormalize predictions and test labels
test_labels_rescaled = denormalize(scaler, test_labels, column_index=0)
predictions_rescaled = denormalize(scaler, predictions[:, 0], column_index=0)

# Evaluate the model
mae = mean_absolute_error(test_labels_rescaled, predictions_rescaled)
r2 = r2_score(test_labels_rescaled, predictions_rescaled)
print(f"Test Results:\n - Mean Absolute Error (MAE): {mae:.4f}\n - R² Score: {r2:.4f}")

# Predict 6 months into the future with 5-minute intervals
six_month_predictions = predict_six_months(
    model=model,
    scaler=scaler,
    dataset=test_dataset,
    start_date=start_date,
    sequence_length=sequence_length,
    steps=8640 # (6 months * 30 days/month * 288 intervals/day) INI KALO MAU BENERAN SIX MONTH DIRUBAH 518 ITU CUMAN PLACEHOLDER, SARANKU 1 MONTH DULU AJA
)

# Save the predictions
six_month_predictions.to_csv("six_month_predictions_5min.csv", index=False)
print("6-month predictions saved to six_month_predictions_5min.csv")

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.plot(test_labels_rescaled[:5000], label="Actual", color='blue')
plt.plot(predictions_rescaled[:5000], label="Predicted", color='orange')
plt.title("LSTM Test Results: Actual vs Predicted")
plt.xlabel("Sample")
plt.ylabel("Uptime Value")
plt.legend()
plt.show()

