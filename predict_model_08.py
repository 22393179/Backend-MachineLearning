from flask import Blueprint, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from utils import load_data, preprocess_data, generate_scatter_plot, train_and_save_model, load_model
import os
import joblib
from sklearn.preprocessing import LabelEncoder

model_08_blueprint = Blueprint('model_08', __name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model_08.pkl')
LE_TARGET_PATH = os.path.join(os.path.dirname(__file__), 'le_target_08.pkl')
FEATURES = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score', 'Conflicts_Over_Social_Media', 'Academic_Level', 'Country']
CATEGORICAL_COLUMNS = ['Academic_Level', 'Country']
NUMERIC_COLUMNS = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score', 'Conflicts_Over_Social_Media']

def train_model_08():
    df = load_data()
    X = df[FEATURES].copy()
    y = df['Relationship_Status']
    le_target = LabelEncoder()
    y_enc = le_target.fit_transform(y)
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        joblib.dump(le, os.path.join(os.path.dirname(__file__), f'le_{col}_08.pkl'))
    model = RandomForestClassifier(random_state=42)
    train_and_save_model(model, X, y_enc, MODEL_PATH)
    joblib.dump(le_target, LE_TARGET_PATH)

@model_08_blueprint.route('/predict-model-08', methods=['POST'])
def predict_model_08():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "error": "No se proporcionaron datos"}), 400

        missing_fields = [field for field in FEATURES if field not in data]
        if missing_fields:
            return jsonify({"status": "error", "error": f"Faltan campos: {', '.join(missing_fields)}"}), 400

        df = load_data()
        input_df, label_encoders = preprocess_data(data, CATEGORICAL_COLUMNS, df)
        for col in NUMERIC_COLUMNS:
            if input_df[col].isna().any():
                return jsonify({"status": "error", "error": f"Valor no numérico en {col}"}), 400

        model = load_model(MODEL_PATH, train_model_08)
        le_target = joblib.load(LE_TARGET_PATH)
        pred_enc = model.predict(input_df[FEATURES])[0]
        proba = float(model.predict_proba(input_df[FEATURES]).max())
        pred_label = le_target.inverse_transform([pred_enc])[0]

        user_value = {'Age': int(data['Age'])}
        plot = generate_scatter_plot(df, 'Age', 'Relationship_Status', pred_enc, user_value)

        return jsonify({
            'status': 'success',
            'predictions': {
                'Relationship_Status': {
                    'value': pred_label,
                    'probability': proba,
                    'interpretation': f"Estado de relación predicho: {pred_label} (probabilidad: {proba:.2%})"
                }
            },
            'plots': {
                'relationship_status_plot': plot
            }
        })

    except ValueError as ve:
        return jsonify({"status": "error", "error": str(ve)}), 400
    except Exception as e:
        return jsonify({"status": "error", "error": f"Error en el servidor: {str(e)}"}), 500