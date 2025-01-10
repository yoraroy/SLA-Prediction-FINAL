# Model Training
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import joblib
import datetime
import sys
from PyQt5.QtWidgets import QApplication, QFileDialog

# Load dataset
def select_test_file():
    app = QApplication(sys.argv)
    file_dialog = QFileDialog()
    file_dialog.setFileMode(QFileDialog.ExistingFile)
    dataset_file, _ = file_dialog.getOpenFileName(
        None, "Select a Test File", ".", "CSV Files (*.csv);;All Files (*)"
    )
    return dataset_file

# Select and load the test dataset
dataset_file = select_test_file()
if not dataset_file:
    print("No file selected. Exiting...")
    exit()

dataset = pd.read_csv(dataset_file)
# Preprocess dataset
dataset['datetime'] = pd.to_datetime(dataset['datetime'], format='%m/%d/%Y %H:%M')
numerical_columns = ['uptimevalue_raw', 'coverage_raw']

scaler = MinMaxScaler()
dataset[numerical_columns] = scaler.fit_transform(dataset[numerical_columns])

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Prepare sequences for LSTM
sequence_length = 6
features = ['uptimevalue_raw', 'coverage_raw']
target = 'uptimevalue_raw'

data = []
labels = []
for i in range(sequence_length, len(dataset)):
    data.append(dataset[features].iloc[i-sequence_length:i].values)
    labels.append(dataset[target].iloc[i])

data = np.array(data)
labels = np.array(labels)

# Split data
train_size = int(0.7 * len(data))
X_train, X_test = data[:train_size], data[train_size:]
y_train, y_test = labels[:train_size], labels[train_size:]

# Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(128, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

# Compile the model with Huber loss
optimizer = Adam(learning_rate=0.001)
huber_loss = tf.keras.losses.Huber()
model.compile(optimizer=optimizer, loss=huber_loss)

# Train the model
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

# Set up early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',    # Monitor validation loss
    patience=10,           # Number of epochs with no improvement after which to stop
    restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored metric
    verbose=1
)

# Train the model with EarlyStopping and TensorBoard callbacks
history = model.fit(
    X_train, y_train, 
    validation_data=(X_test, y_test), 
    epochs=100, 
    batch_size=32, 
    verbose=1, 
    callbacks=[tensorboard_callback, early_stopping]  # Add EarlyStopping here
)

model.evaluate(X_test, y_test)

# Save the model
model.save('lstm_model.h5')
print("Model saved as 'lstm_model.h5'")
