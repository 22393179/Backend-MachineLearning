import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN

class DBSCANModel:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        self.model = None

    def fit(self, df):
        cat_cols = ["Gender", "Academic_Level", "Country", "Most_Used_Platform",
                    "Affects_Academic_Performance", "Relationship_Status"]
        df_encoded = df.copy()
        for col in cat_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            self.label_encoders[col] = le
        
        X = df_encoded.drop(columns=["Student_ID"])
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = DBSCAN(eps=1.5, min_samples=2)
        self.model.fit(X_scaled)

    def preprocess(self, df_new):
        df_new_encoded = df_new.copy()
        for col, le in self.label_encoders.items():
            df_new_encoded[col] = df_new_encoded[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        X_new = df_new_encoded.drop(columns=["Student_ID"], errors='ignore')
        X_new_scaled = self.scaler.transform(X_new)
        return X_new_scaled

    def predict(self, df_new):
        # DBSCAN no tiene predict, así que hacemos fit_predict con los nuevos datos
        X_new_scaled = self.preprocess(df_new)
        return self.model.fit_predict(X_new_scaled)
