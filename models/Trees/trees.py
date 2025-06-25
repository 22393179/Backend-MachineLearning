import matplotlib
matplotlib.use('Agg')
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from io import BytesIO
from pathlib import Path
import base64
import pandas as pd

def get_data_path():
    """Función auxiliar para obtener la ruta correcta del dataset"""
    return Path(__file__).parent.parent.parent / "data" / "Students Social Media Addiction.csv"

# 1. Árbol con variables numéricas (clasificación binaria)
def tree_numeric_binary():

    df = pd.read_csv(get_data_path())

    df['Addiction_High'] = (df['Addicted_Score'] > 7).astype(int)
    
    X = df[['Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night']]
    y = df['Addiction_High']
    
    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X, y)
    
    plt.figure(figsize=(12, 6))
    plot_tree(model, feature_names=X.columns, class_names=['Low', 'High'], filled=True)
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# 2. Árbol con variables categóricas (clasificación binaria)
def tree_categorical_instagram():
    df = pd.read_csv(get_data_path())
    df['Is_Instagram'] = (df['Most_Used_Platform'] == 'Instagram').astype(int)
    
    X = pd.get_dummies(df[['Country', 'Academic_Level']])
    y = df['Is_Instagram']
    
    model = DecisionTreeClassifier(max_depth=4)
    model.fit(X, y)
    
    plt.figure(figsize=(16, 8))
    plot_tree(model, feature_names=X.columns, class_names=['Other', 'Instagram'], filled=True)
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# 3. Árbol mixto (numéricas + categóricas, simulando regularización con max_depth)
def tree_mixed_conflict():
    df = pd.read_csv(get_data_path())
    df['Conflict_High'] = (df['Conflicts_Over_Social_Media'] > 2).astype(int)
    
    X = df[['Avg_Daily_Usage_Hours', 'Mental_Health_Score', 'Relationship_Status']]
    X = pd.get_dummies(X, columns=['Relationship_Status'])
    y = df['Conflict_High']
    
    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X, y)
    
    plt.figure(figsize=(14, 7))
    plot_tree(model, feature_names=X.columns, class_names=['Low', 'High'], filled=True)
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')
