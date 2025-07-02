from flask import Blueprint, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
from utils import load_data, preprocess_data, generate_scatter_plot, train_and_save_model, load_model
import os
import joblib

model_11_blueprint = Blueprint('model_11', __name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model_11.pkl')
FEATURES = ['Age', 'Avg_Daily_Usage_Hours', 'Mental_Health_Score', 'Addicted_Score']
NUMERIC_COLUMNS = FEATURES

def train_model_11():
    df = load_data()
    X = df[FEATURES]
    y = df['Conflicts_Over_Social_Media']
    model = LinearRegression()
    train_and_save_model(model, X, y, MODEL_PATH)

@model_11_blueprint.route('/predict-model-11', methods=['POST'])
def predict_model_11():
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

        model = load_model(MODEL_PATH, train_model_11)
        predicted_value = float(model.predict(input_df[FEATURES])[0])
        predicted_value = max(0, min(predicted_value, 10))

        interpretation = f"Se estiman {predicted_value:.2f} conflictos derivados del uso de redes sociales."
        df = load_data()
        user_value = {'Age': int(data['Age'])}
        plot = generate_scatter_plot(df, 'Age', 'Conflicts_Over_Social_Media', predicted_value, user_value)

        return jsonify({
            'status': 'success',
            'predictions': {
                'Social_Media_Conflicts': {
                    'value': predicted_value,
                    'interpretation': interpretation
                }
            },
            'plots': {
                'social_media_conflicts_plot': plot
            }
        })

    except ValueError as ve:
        return jsonify({"status": "error", "error": str(ve)}), 400
    except Exception as e:
        return jsonify({"status": "error", "error": f"Error en el servidor: {str(e)}"}), 500