import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Cargar datos
data = pd.read_csv('data\Students Social Media Addiction.csv')

# Crear objetivo binario
data['Addicted_Binary'] = data['Addicted_Score'].apply(lambda x: 1 if x >= 7 else 0)

# Seleccionar características y objetivo
features = ['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score']

X = data[features]
y = data['Addicted_Binary']

# Crear características polinómicas (grado=2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)

# Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear y entrenar modelo
modelo = LogisticRegression(max_iter=1000, penalty='l2', C=1.0)
modelo.fit(X_train_scaled, y_train)

# Predecir y evaluar
y_pred = modelo.predict(X_test_scaled)
print("Precisión:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))