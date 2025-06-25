import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Cargar datos
data = pd.read_csv('data\Students Social Media Addiction.csv')

# Crear objetivo categórico (Bajo, Medio, Alto)
data['Addiction_Level'] = pd.cut(data['Addicted_Score'], 
                               bins=[0, 4, 7, 10],
                               labels=['Bajo', 'Medio', 'Alto'])

# Seleccionar características y objetivo
features = ['Age', 'Gender', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 
            'Mental_Health_Score', 'Conflicts_Over_Social_Media']

X = data[features]
y = data['Addiction_Level']

# Convertir variables categóricas
X = pd.get_dummies(X, columns=['Gender'], drop_first=True)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear y entrenar modelo
modelo = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
modelo.fit(X_train_scaled, y_train)

# Predecir y evaluar
y_pred = modelo.predict(X_test_scaled)
print("Precisión:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))