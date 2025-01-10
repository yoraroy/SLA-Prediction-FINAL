import os
from flask import Flask, request, jsonify, send_file
import pandas as pd
from model.preprocessingtest import process_file, calculate_datetime_raw # Import the function from preprocessingtest.py
from model.graphing_prtg import graphing_prtg
from model.graphing_predict import graphing_predict
import sqlite3
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from io import StringIO, BytesIO
from flask_cors import CORS
import shutil
import zipfile

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['PROCESSED_FOLDER'] = './processed'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the SLA Prediction Backend"}), 200

def create_zip(output_dir):
    # Create a ZIP file containing all the PNG files in graphs_prtg
    zip_filename = f"{os.path.basename(output_dir)}.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, output_dir))
    return zip_filename

# Ensure folders exist & fresh
shutil.rmtree(app.config['UPLOAD_FOLDER'], ignore_errors=True); os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
shutil.rmtree(app.config['PROCESSED_FOLDER'], ignore_errors=True); os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
shutil.rmtree("graphs_prtg", ignore_errors=True)
shutil.rmtree("graphs_prtg.zip", ignore_errors=True)
shutil.rmtree("graphs_prediction", ignore_errors=True)
shutil.rmtree("graphs_prediction.zip", ignore_errors=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"message": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "No file selected"}), 400
    if file and allowed_file(file.filename):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'prtg.csv')
        file.save(file_path)
        return jsonify({"message": "File uploaded successfully", "file_path": file_path}), 200

@app.route('/preprocess', methods=['POST'])
def process_file_route():
    uploaded_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'prtg.csv')
    if not os.path.exists(uploaded_file_path):
        return jsonify({"message": "No file to process. Upload a file first."}), 400
    shutil.rmtree("graphs_prtg", ignore_errors=True); os.makedirs("graphs_prtg")
    try:
        # Graphing
        graphing_prtg(uploaded_file_path)
        graph_zip_filename = create_zip("graphs_prtg")
        
        # Process the file
        processed_file_path = process_file(uploaded_file_path)

        # Create a combined ZIP file
        combined_zip_filename = os.path.join(app.config['UPLOAD_FOLDER'], "output_prtg.zip")
        with zipfile.ZipFile(combined_zip_filename, 'w') as combined_zip:
            combined_zip.write(processed_file_path, arcname="output_prtg.csv")
            combined_zip.write(graph_zip_filename, arcname="graphs_prtg.zip")

        return send_file(combined_zip_filename, as_attachment=True, download_name="output_prtg.zip")

    except Exception as e:
        return jsonify({"message": "Error processing file", "error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    # Load uploaded data
    processed_file = './uploads/prtg.csv'
    data = pd.read_csv(processed_file)

    # Preprocess data for LSTM
    scaler = joblib.load('./model/scaler.pkl')
    numerical_columns = ['uptimevalue_raw', 'coverage_raw']
    data[numerical_columns] = scaler.transform(data[numerical_columns])

    sequence_length = 6
    features = ['uptimevalue_raw', 'coverage_raw']
    sequences = [
        data[features].iloc[i - sequence_length:i].values
        for i in range(sequence_length, len(data))
    ]
    sequences = np.array(sequences)

    # Load model and make predictions
    model = load_model('./model/lstm_model.h5')
    predictions = model.predict(sequences).flatten()

    # Denormalize predictions
    predictions_rescaled = scaler.inverse_transform([[p, 0] for p in predictions])[:, 0]

    # Save predictions to CSV
    predictions_file_path = './processed/predictions_data.csv'
    data['datetime'] = pd.to_datetime(data['datetime'], format='%m/%d/%Y %H:%M')  # Convert datetime to correct format
    prediction_df = pd.DataFrame({
        'datetime': data['datetime'].iloc[sequence_length:],
        'datetime_raw': data['datetime_raw'].iloc[sequence_length:],
        'uptimevalue_raw': predictions_rescaled,
        'coverage_raw': data['coverage_raw'].iloc[sequence_length:],
    })
    prediction_df.to_csv(predictions_file_path, index=False)

    # Load the predictions data
    predictions_data = pd.read_csv(predictions_file_path)

    # Convert 'datetime' to the desired format
    predictions_data['datetime'] = pd.to_datetime(predictions_data['datetime']).dt.strftime('%m/%d/%Y %H:%M')

    # Save the updated predictions file
    predictions_data.to_csv(predictions_file_path, index=False)


    shutil.rmtree("graphs_prediction", ignore_errors=True); os.makedirs("graphs_prediction")
    # Call the graphing function with the updated file
    graphing_predict(predictions_file_path)

    # Create ZIP of graphs
    zip_file_path = create_zip("graphs_prediction")

    # Create a combined ZIP file
    combined_zip_filename = os.path.join(app.config['UPLOAD_FOLDER'], "output_predictions.zip")
    with zipfile.ZipFile(combined_zip_filename, 'w') as combined_zip:
        combined_zip.write(predictions_file_path, arcname="output_prediction.csv")
        combined_zip.write(zip_file_path, arcname="graphs_prediction.zip")

    return send_file(combined_zip_filename, as_attachment=True, download_name="output_predictions.zip")

@app.route('/graphs', methods=['GET'])
def get_graphs():
    output_dir = "graphs_prtg"
    images = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.jpg')]
    return jsonify(images)  # Returning list of image paths or URLs

# Enable CORS on the Flask app
CORS(app, resources={r"/*": {"origins": "*"}})

if __name__ == '__main__':
    app.run(debug=True)
