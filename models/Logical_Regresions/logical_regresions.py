from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd

def logistic_binary():
    df = pd.read_csv("../../data/Students Social Media Addiction.csv")
    
    # Binzarizar la variable objetivo
    df['Addiction_High'] = (df['Addicted_Score'] > 7).astype(int)
    
    X = df[['Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night']]
    y = df['Addiction_High']
    
    model = LogisticRegression()
    model.fit(X, y)
    
    # Gráfico de frontera de decisión
    plt.figure(figsize=(10, 6))
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap='coolwarm', alpha=0.6)
    plt.title('Clasificación Binaria: Adicción Alta vs. Baja')
    plt.xlabel('Horas de Uso Diario')
    plt.ylabel('Horas de Sueño')
    
    # Convertir a base64
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def logistic_multiclass():
    df = pd.read_csv("../../data/Students Social Media Addiction.csv")
    
    # Crear categorías
    df['Addiction_Level'] = pd.cut(df['Addicted_Score'], 
                                  bins=[0, 4, 7, 10], 
                                  labels=['Bajo', 'Medio', 'Alto'])
    
    X = df[['Avg_Daily_Usage_Hours', 'Age', 'Mental_Health_Score']]
    y = df['Addiction_Level']
    
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    model.fit(X, y)
    
    # Matriz de confusión (ejemplo simplificado)
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
    disp.plot(cmap='Blues')
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def logistic_regularized():
    df = pd.read_csv("../../data/Students Social Media Addiction.csv")
    
    X = df[['Avg_Daily_Usage_Hours', 'Addicted_Score', 'Relationship_Status']]
    X = pd.get_dummies(X, columns=['Relationship_Status'])  # One-hot encoding
    y = (df['Conflicts_Over_Social_Media'] > 2).astype(int)  # ¿Conflictos frecuentes?
    
    model = LogisticRegression(penalty='l2', C=0.1, solver='liblinear')  # Regularización fuerte
    model.fit(X, y)
    
    # Importancia de características
    plt.barh(X.columns, model.coef_[0])
    plt.title('Coeficientes del Modelo (Regularización L2)')
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def logistic_platform():
    df = pd.read_csv("../../data/Students Social Media Addiction.csv")
    
    X = df[['Age', 'Addicted_Score', 'Avg_Daily_Usage_Hours']]
    y = (df['Most_Used_Platform'] == 'Instagram').astype(int)
    
    model = LogisticRegression()
    model.fit(X, y)
    
    # Curva ROC
    from sklearn.metrics import roc_curve, roc_auc_score
    y_prob = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y, y_prob):.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Curva ROC: Preferencia por Instagram')
    plt.legend()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')


img_binary = logistic_binary()
with open("binary.png", "wb") as f:
    f.write(base64.b64decode(img_binary))