import os
import json
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_and_save():
    current_dir = Path(__file__).parent
    csv_path = current_dir.parent / "data" / "Students_Social_Media_Addiction.csv"
    models_dir = current_dir.parent / "models" / "saved_models"
    
    df = pd.read_csv(csv_path)
    X = df.drop(["Is_Addicted", "Addicted_Score"], axis=1)
    X = pd.get_dummies(X)
    y = df["Is_Addicted"]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, models_dir / "random_forest.pkl")

if __name__ == "__main__":
    train_and_save()