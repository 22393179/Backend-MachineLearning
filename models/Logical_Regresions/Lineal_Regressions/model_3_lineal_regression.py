import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

def train_model_3():
    df = pd.read_csv("data/Students Social Media Addiction.csv")
    X = df[["Conflicts_Over_Social_Media", "Mental_Health_Score"]]
    y = df["Addicted_Score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, "saved_models/model_3.pkl")
    print("✅ Modelo 3 guardado como model_3.pkl")