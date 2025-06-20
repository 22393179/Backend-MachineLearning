import os
import json
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import joblib

def train_and_save():
    # Configurar rutas
    current_dir = Path(__file__).parent
    csv_path = current_dir.parent / "data" / "Students_Social_Media_Addiction.csv"
    models_dir = current_dir.parent / "models" / "saved_models"
    
    # 1. Cargar datos
    df = pd.read_csv(csv_path)
    X = df.drop(["Is_Addicted", "Addicted_Score"], axis=1)
    X = pd.get_dummies(X)
    y = df["Is_Addicted"]
    
    # 2. Entrenar modelo
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    # 3. Guardar modelo
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, models_dir / "logistic_regression.pkl")
    
    # 4. Guardar columnas (solo si no existe)
    columns_file = current_dir.parent / "models" / "columns.json"
    if not columns_file.exists():
        with open(columns_file, "w") as f:
            json.dump({"features": X.columns.tolist()}, f, indent=4)

if __name__ == "__main__":
    train_and_save()