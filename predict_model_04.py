from flask import Blueprint, request, jsonify
import pandas as pd
from sklearn.linear_model import LogisticRegression
from utils import load_data, preprocess_data, generate_scatter_plot, train_and_save_model, load_model
import os
import joblib
from sklearn.preprocessing import LabelEncoder

model_04_blueprint = Blueprint('model_04', __name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model_04.pkl')
FEATURES = [
    'Age', 'Gender', 'Academic_Level', 'Country', 'Avg_Daily_Usage_Hours',
    'Most_Used_Platform', 'Sleep_Hours_Per_Night', 'Mental_Health_Score',
    'Relationship_Status', 'Conflicts_Over_Social_Media', 'Addicted_Score'
]
CATEGORICAL_COLUMNS = ['Gender', 'Academic_Level', 'Country', 'Most_Used_Platform', 'Relationship_Status']
NUMERIC_COLUMNS = [
    'Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night',
    'Mental_Health_Score', 'Conflicts_Over_Social_Media', 'Addicted_Score'
]

def train_model_04():
    df = load_data()
    X = df[FEATURES].copy()
    y = df['Affects_Academic_Performance']
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        joblib.dump(le, os.path.join(os.path.dirname(__file__), f'le_{col}_04.pkl'))
    model = LogisticRegression(max_iter=1000)
    train_and_save_model(model, X, y, MODEL_PATH)

@model_04_blueprint.route('/predict-model-04', methods=['POST'])
def predict_model_04():
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

        model = load_model(MODEL_PATH, train_model_04)
        X_input = input_df[FEATURES]
        predicted_class = int(model.predict(X_input)[0])
        predicted_proba = float(model.predict_proba(X_input)[0][1])

        interpretation = "Es probable que las redes sociales afecten negativamente el rendimiento académico." if predicted_class == 1 else "No se estima un impacto académico negativo por redes sociales."
        interpretation += f" Probabilidad estimada: {predicted_proba:.2%}"
        user_value = {'Age': int(data['Age'])}
        plot = generate_scatter_plot(df, 'Age', 'Affects_Academic_Performance', predicted_class, user_value)

        return jsonify({
            'status': 'success',
            'predictions': {
                'Academic_Impact_Prediction': {
                    'value': predicted_class,
                    'probability': predicted_proba,
                    'interpretation': interpretation
                }
            },
            'plots': {
                'academic_impact_prediction_plot': plot
            }
        })

    except ValueError as ve:
        return jsonify({"status": "error", "error": str(ve)}), 400
    except Exception as e:
        return jsonify({"status": "error", "error": f"Error en el servidor: {str(e)}"}), 500