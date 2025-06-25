import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

def train_model_4():
    df = pd.read_csv("data/Students Social Media Addiction.csv")
    df = df.drop(columns=["Student_ID", "Addicted_Score"])
    df = pd.get_dummies(df)

    X = df
    y = pd.read_csv("data/Students Social Media Addiction.csv")["Addicted_Score"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, "saved_models/model_4.pkl")
    print("✅ Modelo 4 guardado como model_4.pkl")