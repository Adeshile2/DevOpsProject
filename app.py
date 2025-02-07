from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Load trained model and encoder
model = joblib.load("ml_model.pkl")
encoder = joblib.load("encoder.pkl")

# Define feature names
feature_columns = ['Age', 'Fare', 'Pclass', 'Sex', 'Embarked']

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        data = request.get_json()

        # Convert input to DataFrame
        df = pd.DataFrame([data], columns=feature_columns)

        # Transform categorical features
        transformed_data = encoder.transform(df)

        # Predict using trained model
        prediction = model.predict(transformed_data)

        # Return result
        return jsonify({'Survived': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
