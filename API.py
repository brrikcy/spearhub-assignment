from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import os

app = Flask(__name__)

# Initialize global variables
lr_model = None
dt_model = None
data = None


@app.route('/upload', methods=['POST'])
def upload_data():
    """
    Upload CSV file containing manufacturing data.
    """
    global data
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request!"}), 400

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "Uploaded file must be a CSV!"}), 400

    try:
        data = pd.read_csv(file)
        data = data.drop_duplicates()
        data = data.dropna()
        return jsonify({"message": "Data uploaded successfully!", "rows": len(data)}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to process file: {str(e)}"}), 500


@app.route('/train', methods=['POST'])
def train_model():
    """
    Train Logistic Regression and Decision Tree models on uploaded data.
    """
    global lr_model, dt_model, data
    if data is None:
        return jsonify({"error": "No data uploaded! Please upload data first using /upload."}), 400

    try:
        # Split data into features (X) and target (y)
        X = data.drop(columns=['Machine_ID', 'Downtime_Flag'])
        y = data['Downtime_Flag']

        # Train-test split
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Logistic Regression model
        lr_model = LogisticRegression()
        lr_model.fit(x_train, y_train)

        # Train Decision Tree model
        dt_model = DecisionTreeClassifier()
        dt_model.fit(x_train, y_train)

        # Save models for future use
        joblib.dump(lr_model, 'lr_model.pkl')
        joblib.dump(dt_model, 'dt_model.pkl')

        # Evaluate models
        lr_report = classification_report(y_test, lr_model.predict(x_test), output_dict=True)
        dt_report = classification_report(y_test, dt_model.predict(x_test), output_dict=True)

        return jsonify({
            "Logistic_Regression_Report": lr_report,
            "Decision_Tree_Report": dt_report
        }), 200
    except Exception as e:
        return jsonify({"error": f"Training failed: {str(e)}"}), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Ensure the input features match the trained model
        required_features = ['Temperature', 'Run_Time', 'Vibration_Level']

        # Check for missing features and set default values
        for feature in required_features:
            if feature not in data:
                return jsonify({
                    "error": f"Missing feature: {feature}. Please provide values for all required features: {required_features}"
                }), 400

        # Convert input data to DataFrame
        input_data = pd.DataFrame([data])

        # Reorder columns to match training data
        input_data = input_data[required_features]

        # Make predictions with both models
        lr_prediction = lr_model.predict(input_data)[0]
        lr_confidence = max(lr_model.predict_proba(input_data)[0])

        dt_prediction = dt_model.predict(input_data)[0]
        dt_confidence = max(dt_model.predict_proba(input_data)[0])

        # Convert predictions to human-readable format
        result = {
            "Logistic_Regression": {
                "Prediction": "Yes" if lr_prediction == 1 else "No",
                "Confidence": round(lr_confidence, 2)
            },
            "Decision_Tree": {
                "Prediction": "Yes" if dt_prediction == 1 else "No",
                "Confidence": round(dt_confidence, 2)
            }
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.errorhandler(Exception)
def handle_exception(e):
    """
    General error handler for unexpected exceptions.
    """
    return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Clean up existing model files (if any) on each restart
    if os.path.exists('lr_model.pkl'):
        os.remove('lr_model.pkl')
    if os.path.exists('dt_model.pkl'):
        os.remove('dt_model.pkl')

    app.run(debug=True)
