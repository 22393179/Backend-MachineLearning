from flask import Blueprint, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
from utils import load_data, preprocess_data, generate_scatter_plot, train_and_save_model, load_model
import os
import joblib

model_10_blueprint = Blueprint('model_10', __name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'modelos', 'model_10.pkl')
FEATURES = ['Addicted_Score']
NUMERIC_COLUMNS = FEATURES

def train_model_10():
    df = load_data()
    X = df[FEATURES]
    y = df['Sleep_Hours_Per_Night']
    model = LinearRegression()
    train_and_save_model(model, X, y, MODEL_PATH)

@model_10_blueprint.route('/predict-model-10', methods=['POST'])
def predict_model_10():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "error": "No se proporcionaron datos"}), 400

        missing_fields = [field for field in FEATURES if field not in data]
        if missing_fields:
            return jsonify({"status": "error", "error": f"Faltan campos: {', '.join(missing_fields)}"}), 400

        input_df, _ = preprocess_data(data, [], df=None)
        for col in NUMERIC_COLUMNS:
            if input_df[col].isna().any():
                return jsonify({"status": "error", "error": f"Valor no numérico en {col}"}), 400

        model = load_model(MODEL_PATH, train_model_10)
        predicted_value = float(model.predict(input_df[FEATURES])[0])
        predicted_value = max(0, min(predicted_value, 24))

        interpretation = f"Se estiman {predicted_value:.2f} horas de sueño basadas en el nivel de adicción."
        df = load_data()
        user_value = {'Addicted_Score': float(data['Addicted_Score'])}
        plot = generate_scatter_plot(df, 'Addicted_Score', 'Sleep_Hours_Per_Night', predicted_value, user_value)

        return jsonify({
            'status': 'success',
            'predictions': {
                'Addiction_vs_Sleep': {
                    'value': predicted_value,
                    'interpretation': interpretation
                }
            },
            'plots': {
                'addiction_vs_sleep_plot': plot
            }
        })

    except ValueError as ve:
        return jsonify({"status": "error", "error": str(ve)}), 400
    except Exception as e:
        return jsonify({"status": "error", "error": f"Error en el servidor: {str(e)}"}), 500