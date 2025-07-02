from flask import Blueprint, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
from utils import load_data, preprocess_data, generate_scatter_plot, train_and_save_model, load_model
import os
import joblib
from sklearn.preprocessing import LabelEncoder

model_03_blueprint = Blueprint('model_03', __name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model_03.pkl')
FEATURES = [
    'Age', 'Gender', 'Academic_Level', 'Country', 'Avg_Daily_Usage_Hours',
    'Most_Used_Platform', 'Sleep_Hours_Per_Night', 'Mental_Health_Score',
    'Relationship_Status', 'Addicted_Score', 'Affects_Academic_Performance'
]
CATEGORICAL_COLUMNS = ['Gender', 'Academic_Level', 'Country', 'Most_Used_Platform', 'Relationship_Status']
NUMERIC_COLUMNS = [
    'Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night',
    'Mental_Health_Score', 'Addicted_Score', 'Affects_Academic_Performance'
]

def train_model_03():
    df = load_data()
    X = df[FEATURES].copy()
    y = df['Conflicts_Over_Social_Media']
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        joblib.dump(le, os.path.join(os.path.dirname(__file__), f'le_{col}_03.pkl'))
    model = LinearRegression()
    train_and_save_model(model, X, y, MODEL_PATH)

@model_03_blueprint.route('/predict-model-03', methods=['POST'])
def predict_model_03():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "error": "No se proporcionaron datos"}), 400

        missing_fields = [field for field in FEATURES if field not in data]
        if missing_fields:
            return jsonify({"status": "error", "error": f"Faltan campos: {', '.join(missing_fields)}"}), 400

        df = load_data()
        input_df, _ = preprocess_data(data, CATEGORICAL_COLUMNS, df)
        for col in NUMERIC_COLUMNS:
            if input_df[col].isna().any():
                return jsonify({"status": "error", "error": f"Valor no numérico en {col}"}), 400

        model = load_model(MODEL_PATH, train_model_03)
        X_input = input_df[FEATURES]
        predicted_value = float(model.predict(X_input)[0])
        predicted_value = max(0, min(predicted_value, 10))

        interpretation = f"Se estiman {predicted_value:.2f} conflictos derivados del uso de redes sociales."
        user_value = {'Age': int(data['Age'])}
        plot = generate_scatter_plot(df, 'Age', 'Conflicts_Over_Social_Media', predicted_value, user_value)

        return jsonify({
            'status': 'success',
            'predictions': {
                'Conflicts_Prediction': {
                    'value': predicted_value,
                    'interpretation': interpretation
                }
            },
            'plots': {
                'conflicts_prediction_plot': plot
            }
        })

    except ValueError as ve:
        return jsonify({"status": "error", "error": str(ve)}), 400
    except Exception as e:
        return jsonify({"status": "error", "error": f"Error en el servidor: {str(e)}"}), 500