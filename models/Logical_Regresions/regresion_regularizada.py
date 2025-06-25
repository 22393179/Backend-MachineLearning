import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Cargar datos
data = pd.read_csv('data\Students Social Media Addiction.csv')


# Crear objetivo binario
data['Addicted_Binary'] = data['Addicted_Score'].apply(lambda x: 1 if x >= 7 else 0)

# Seleccionar características y objetivo
features = ['Age', 'Gender', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 
           'Mental_Health_Score', 'Conflicts_Over_Social_Media', 'Affects_Academic_Performance']

X = data[features]
y = data['Addicted_Binary']

# Convertir variables categóricas
X = pd.get_dummies(X, columns=['Gender', 'Affects_Academic_Performance'], drop_first=True)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ajuste de hiperparámetros con regularización
param_grid = {
    'C': np.logspace(-4, 4, 20),
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']  # liblinear funciona con L1 y L2
}

# Crear y entrenar modelo con búsqueda en cuadrícula
modelo = GridSearchCV(LogisticRegression(max_iter=1000), 
                     param_grid, 
                     cv=5, 
                     scoring='accuracy')
modelo.fit(X_train_scaled, y_train)

# Mejores parámetros
print("Mejores parámetros:", modelo.best_params_)

# Predecir y evaluar con el mejor modelo
mejor_modelo = modelo.best_estimator_
y_pred = mejor_modelo.predict(X_test_scaled)
print("Precisión:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Importancia de características
importancia = pd.DataFrame({'Característica': X.columns, 'Importancia': mejor_modelo.coef_[0]})
print(importancia.sort_values('Importancia', ascending=False))