from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import json
from pathlib import Path

app = FastAPI()

# Configurar rutas
current_dir = Path(__file__).parent
models_dir = current_dir / "models" / "saved_models"
columns_file = current_dir / "models" / "columns.json"

# Cargar configuración
with open(columns_file) as f:
    FEATURES = json.load(f)["features"]

# Cargar modelos
MODELS = {
    "logistic": joblib.load(models_dir / "logistic_regression.pkl"),
    "random_forest": joblib.load(models_dir / "random_forest.pkl"),
    "knn": joblib.load(models_dir / "knn.pkl")
}

@app.post("/predict/{model_name}")
async def predict(model_name: str, data: dict):
    if model_name not in MODELS:
        raise HTTPException(status_code=404, detail="Modelo no encontrado")
    
    try:
        # Preprocesamiento
        df = pd.DataFrame([data])
        df = pd.get_dummies(df)
        
        # Asegurar columnas
        for feature in FEATURES:
            if feature not in df.columns:
                df[feature] = 0
        df = df[FEATURES]
        
        # Predecir
        prediction = int(MODELS[model_name].predict(df)[0])
        probability = float(MODELS[model_name].predict_proba(df)[0][1])
        
        return {
            "prediction": prediction,
            "probability": probability,
            "model": model_name
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))