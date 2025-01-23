# Manufacturing Downtime Prediction API

This repository contains a Flask application (`API.ipynb`) that serves as a web API for predicting machine downtime in a manufacturing setting. It leverages Logistic Regression and Decision Tree models for predictions.

## Project Structure

- `API.ipynb`: The Jupyter Notebook file containing the main Flask application code. It defines the API endpoints for data upload, model training, prediction, and error handling.
- `README.md` (current file): Provides an overview of the project, including API functionality, testing instructions, and dataset details.
- `screenshots` : Contain screenshots showcasing the API tested using Postman.
- `machine_downtime.csv` : The CSV dataset used for training and testing the machine downtime prediction models.

## API Endpoints

### **/upload (POST)**

- Uploads a CSV file containing manufacturing data.
- **Expected format**: CSV with columns including `Machine_ID`,`Temperature`,`Runtime`, `Downtime_Flag` (target variable), and other relevant features.
- **Response**:
  - Success: JSON with message "Data uploaded successfully!" and number of rows uploaded.
  - Error: JSON with error message (e.g., file not found, invalid file format).

### **/train (POST)**

- Trains Logistic Regression and Decision Tree models on the uploaded data.
- **Prerequisite**: Data must be uploaded first using the `/upload` endpoint.
- **Response**:
  - Success: JSON containing classification reports for both models.
  - Error: JSON with error message (e.g., no data uploaded, training failure).

### **/predict (POST)**

- Predicts machine downtime using the trained models for a given set of input features.
- **Required data in JSON format**:
  - `Temperature` (float)
  - `Run_Time` (float)
  - `Vibration_Level` (float)
- **Response**:
  - Success: JSON with predictions and confidence scores for both Logistic Regression and Decision Tree models.
  - Error: JSON with error message (e.g., missing features, prediction failure).

## Running the Application

1. Ensure you have Python and the required libraries installed.
2. Open the `API.ipynb` file in a Jupyter Notebook environment.
3. Run the code cells in the notebook. The API will typically start running on `http://127.0.0.1:5000/` (default Flask development server) by default.

## Testing the API with Postman

1. Install Postman ([Postman Download](https://www.postman.com/)) - a popular tool for testing APIs.
2. Create a new POST request in Postman.
3. Set the URL to `http://127.0.0.1:5000/<endpoint>`, replacing `<endpoint>` with the desired endpoint (e.g., `/upload`, `/train`, `/predict`).
4. For file uploads (`/upload`), use the "File" tab in Postman to select the CSV file.
5. For predictions (`/predict`), add a JSON payload to the request body containing the features for which you want a prediction (e.g., `{"Temperature": 50.5, "Run_Time": 200, "Vibration_Level": 1.2}`).
6. Send the request and view the response. Refer to the API Endpoints section for expected responses.

## Dataset

The `machine_downtime.csv` file provides sample data used for training and testing the prediction models. It should contain relevant features that can influence machine downtime.
