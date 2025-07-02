from flask import Blueprint, request, jsonify
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from utils import load_data, preprocess_data, generate_scatter_plot, train_and_save_model, load_model
import os
import joblib

model_05_blueprint = Blueprint('model_05', __name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model_05.pkl')
FEATURES = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Conflicts_Over_Social_Media', 'Addicted_Score', 'Academic_Level']
NUMERIC_COLUMNS = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Conflicts_Over_Social_Media', 'Addicted_Score']
CATEGORICAL_COLUMNS = ['Academic_Level']

def train_model_05():
    df = load_data()
    df['Mental_Health_Risk'] = (df['Mental_Health_Score'] <= 5).astype(int)  # 1 = riesgo, 0 = sin riesgo
    X = df[FEATURES]
    y = df['Mental_Health_Risk']
    
    # Preprocesar datos de entrenamiento
    X_processed, encoders = preprocess_data(X, categorical_columns=CATEGORICAL_COLUMNS, df=df)
    model = RandomForestClassifier(random_state=42)
    train_and_save_model(model, X_processed, y, MODEL_PATH)
    
    # Guardar el encoder para su uso en predicción
    joblib.dump(encoders, MODEL_PATH.replace('.pkl', '_encoders.pkl'))

@model_05_blueprint.route('/predict-model-05', methods=['POST'])
def predict_model_05():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "error": "No se proporcionaron datos"}), 400

        print(f"Datos recibidos: {data}")  # Debugging

        missing_fields = [field for field in FEATURES if field not in data]
        if missing_fields:
            return jsonify({"status": "error", "error": f"Faltan campos: {', '.join(missing_fields)}"}), 400

        # Preprocesar datos de entrada
        input_df, _ = preprocess_data(data, categorical_columns=CATEGORICAL_COLUMNS, df=load_data())
        print(f"input_df después de preprocess: {input_df.to_dict()}")  # Debugging
        for col in NUMERIC_COLUMNS:
            if input_df[col].isna().any():
                return jsonify({"status": "error", "error": f"Valor no numérico en {col}"}), 400

        # Verificar forma del DataFrame
        if input_df.ndim != 2:
            return jsonify({"status": "error", "error": f"Input debe ser 2D, pero tiene forma: {input_df.shape}"}), 400

        model = load_model(MODEL_PATH, train_model_05)
        prediction = int(model.predict(input_df[FEATURES])[0])
        proba = float(model.predict_proba(input_df[FEATURES])[0][1])

        interpretation = "Riesgo de salud mental baja." if prediction == 0 else "Sin riesgo de salud mental baja."
        interpretation += f" Probabilidad de riesgo: {proba:.2%}"

        df = load_data()
        user_value = {'Avg_Daily_Usage_Hours': float(data['Avg_Daily_Usage_Hours'])}
        plot = generate_scatter_plot(df, 'Avg_Daily_Usage_Hours', 'Mental_Health_Score', proba, user_value)

        return jsonify({
            'status': 'success',
            'predictions': {
                'Mental_Health_Risk_Prediction': {
                    'value': prediction,
                    'probability': proba,
                    'interpretation': interpretation
                }
            },
            'plots': {
                'mental_health_risk_prediction_plot': plot
            }
        }), 200

    except ValueError as ve:
        return jsonify({"status": "error", "error": f"Valor no válido: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"status": "error", "error": f"Error en el servidor: {str(e)}"}), 500