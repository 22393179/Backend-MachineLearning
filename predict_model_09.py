from flask import Blueprint, request, jsonify
import pandas as pd
from sklearn.cluster import KMeans
from utils import load_data, preprocess_data, generate_scatter_plot, train_and_save_model, load_model
import os
import joblib

model_09_blueprint = Blueprint('model_09', __name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'modelos', 'model_09.pkl')
FEATURES = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score', 'Conflicts_Over_Social_Media', 'Addicted_Score']
NUMERIC_COLUMNS = FEATURES

def train_model_09():
    df = load_data()
    X = df[FEATURES]
    model = KMeans(n_clusters=3, random_state=42)
    train_and_save_model(model, X, None, MODEL_PATH)  # KMeans no usa y
    return model

@model_09_blueprint.route('/predict-model-09', methods=['POST'])
def predict_model_09():
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

        model = load_model(MODEL_PATH, train_model_09)
        cluster = int(model.predict(input_df[FEATURES])[0])

        df = load_data()
        user_value = {'Age': int(data['Age'])}
        plot = generate_scatter_plot(df, 'Age', 'Addicted_Score', cluster, user_value)

        return jsonify({
            'status': 'success',
            'predictions': {
                'Cluster': {
                    'value': cluster,
                    'interpretation': f"Usuario asignado al clúster {cluster}."
                }
            },
            'plots': {
                'cluster_plot': plot
            }
        })

    except ValueError as ve:
        return jsonify({"status": "error", "error": str(ve)}), 400
    except Exception as e:
        return jsonify({"status": "error", "error": f"Error en el servidor: {str(e)}"}), 500