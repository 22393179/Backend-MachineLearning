from flask import Blueprint, request, jsonify
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from utils import load_data, preprocess_data, generate_scatter_plot, train_and_save_model, load_model
import os
import joblib

model_06_blueprint = Blueprint('model_06', __name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'modelos', 'model_06.pkl')
FEATURES = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Conflicts_Over_Social_Media', 'Mental_Health_Score']
NUMERIC_COLUMNS = FEATURES

def train_model_06():
    df = load_data()
    df['High_Addiction'] = (df['Addicted_Score'] >= 7).astype(int)
    X = df[FEATURES]
    y = df['High_Addiction']
    model = RandomForestClassifier(random_state=42)
    train_and_save_model(model, X, y, MODEL_PATH)

@model_06_blueprint.route('/predict-model-06', methods=['POST'])
def predict_model_06():
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

        model = load_model(MODEL_PATH, train_model_06)
        prediction = int(model.predict(input_df[FEATURES])[0])
        proba = float(model.predict_proba(input_df[FEATURES])[0][1])

        interpretation = "Alta adicción a redes sociales." if prediction == 1 else "Baja adicción a redes sociales."
        interpretation += f" Probabilidad de adicción alta: {proba:.2%}"

        df = load_data()
        user_value = {'Avg_Daily_Usage_Hours': float(data['Avg_Daily_Usage_Hours'])}
        plot = generate_scatter_plot(df, 'Avg_Daily_Usage_Hours', 'Addicted_Score', proba, user_value)

        return jsonify({
            'status': 'success',
            'predictions': {
                'Addiction_Classification': {
                    'value': prediction,
                    'probability': proba,
                    'interpretation': interpretation
                }
            },
            'plots': {
                'addiction_classification_plot': plot
            }
        })

    except ValueError as ve:
        return jsonify({"status": "error", "error": str(ve)}), 400
    except Exception as e:
        return jsonify({"status": "error", "error": f"Error en el servidor: {str(e)}"}), 500