from flask import Blueprint, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from utils import load_data, preprocess_data, generate_scatter_plot, train_and_save_model, load_model
import os
import joblib

model_01_blueprint = Blueprint('model_01', __name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model_01.pkl')
FEATURES = [
    'Age', 'Gender', 'Academic_Level', 'Country', 'Avg_Daily_Usage_Hours',
    'Most_Used_Platform', 'Sleep_Hours_Per_Night', 'Mental_Health_Score',
    'Relationship_Status', 'Conflicts_Over_Social_Media', 'Affects_Academic_Performance'
]
CATEGORICAL_COLUMNS = ['Gender', 'Academic_Level', 'Country', 'Most_Used_Platform', 'Relationship_Status']
NUMERIC_COLUMNS = [
    'Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night',
    'Mental_Health_Score', 'Conflicts_Over_Social_Media', 'Affects_Academic_Performance'
]

def train_model_01():
    df = load_data()
    X = df[FEATURES].copy()
    y = df['Addicted_Score']
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        joblib.dump(le, os.path.join(os.path.dirname(__file__), f'le_{col}_01.pkl'))
    model = LinearRegression()
    train_and_save_model(model, X, y, MODEL_PATH)

@model_01_blueprint.route('/predict-model-01', methods=['POST'])
def predict_model_01():
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

        model = load_model(MODEL_PATH, train_model_01)
        X_input = input_df[FEATURES]
        predicted_value = float(model.predict(X_input)[0])
        predicted_value = max(0, min(predicted_value, 10))

        interpretation = f"Nivel de adicción estimado: {predicted_value:.2f} en escala 0-10."
        user_value = {'Age': int(data['Age'])}
        plot = generate_scatter_plot(df, 'Age', 'Addicted_Score', predicted_value, user_value)

        return jsonify({
            'status': 'success',
            'predictions': {
                'Addicted_Score_Prediction': {
                    'value': predicted_value,
                    'interpretation': interpretation
                }
            },
            'plots': {
                'addicted_score_prediction_plot': plot
            }
        })

    except ValueError as ve:
        return jsonify({"status": "error", "error": str(ve)}), 400
    except Exception as e:
        return jsonify({"status": "error", "error": f"Error en el servidor: {str(e)}"}), 500