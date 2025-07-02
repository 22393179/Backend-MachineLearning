from flask import Blueprint, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from utils import load_data, preprocess_data, generate_scatter_plot, train_and_save_model, load_model
import os
import joblib

model_12_blueprint = Blueprint('model_12', __name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'modelos', 'model_12.pkl')
FEATURES = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score', 'Addicted_Score']
NUMERIC_COLUMNS = FEATURES

def train_model_12():
    df = load_data()
    df['Academic_Performance_Risk'] = (df['Affects_Academic_Performance'] == 1).astype(int)
    X = df[FEATURES]
    y = df['Academic_Performance_Risk']
    model = RandomForestClassifier(random_state=42)
    train_and_save_model(model, X, y, MODEL_PATH)

@model_12_blueprint.route('/predict-model-12', methods=['POST'])
def predict_model_12():
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

        model = load_model(MODEL_PATH, train_model_12)
        prediction = int(model.predict(input_df[FEATURES])[0])
        proba = float(model.predict_proba(input_df[FEATURES])[0][1])

        interpretation = "Riesgo de bajo rendimiento académico." if prediction == 1 else "Sin riesgo de bajo rendimiento académico."
        interpretation += f" Probabilidad de riesgo: {proba:.2%}"

        df = load_data()
        user_value = {'Age': int(data['Age'])}
        plot = generate_scatter_plot(df, 'Age', 'Affects_Academic_Performance', prediction, user_value)

        return jsonify({
            'status': 'success',
            'predictions': {
                'Academic_Performance_Risk': {
                    'value': prediction,
                    'probability': proba,
                    'interpretation': interpretation
                }
            },
            'plots': {
                'academic_performance_risk_plot': plot
            }
        })

    except ValueError as ve:
        return jsonify({"status": "error", "error": str(ve)}), 400
    except Exception as e:
        return jsonify({"status": "error", "error": f"Error en el servidor: {str(e)}"}), 500