import os
import pandas as pd
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
import joblib

def train_and_save():
    current_dir = Path(__file__).parent
    csv_path = current_dir.parent / "data" / "Students_Social_Media_Addiction.csv"
    models_dir = current_dir.parent / "models" / "saved_models"
    
    df = pd.read_csv(csv_path)
    X = df.drop(["Is_Addicted", "Addicted_Score"], axis=1)
    X = pd.get_dummies(X)
    y = df["Is_Addicted"]
    
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)
    
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, models_dir / "knn.pkl")

if __name__ == "__main__":
    train_and_save()