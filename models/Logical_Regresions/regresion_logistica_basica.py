import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Cargar datos
data = pd.read_csv('data\Students Social Media Addiction.csv')

# Crear objetivo binario (1 si Addicted_Score >= 7, else 0)
data['Addicted_Binary'] = data['Addicted_Score'].apply(lambda x: 1 if x >= 7 else 0)

# Seleccionar características y objetivo
features = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score', 'Conflicts_Over_Social_Media']

X = data[features]
y = data['Addicted_Binary']

# Convertir variables categóricas (Género)
X = pd.get_dummies(X, columns=['Gender'], drop_first=True)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear y entrenar modelo
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train_scaled, y_train)

# Predecir y evaluar
y_pred = modelo.predict(X_test_scaled)
print("Precisión:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Importancia de características
importancia = pd.DataFrame({'Característica': X.columns, 'Importancia': modelo.coef_[0]})
print(importancia.sort_values('Importancia', ascending=False))