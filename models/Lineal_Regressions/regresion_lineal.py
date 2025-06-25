from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd

def Predicting_the_level_of_social_media_addiction(debug=False):
    try:
        df = pd.read_csv("../../data/Students Social Media Addiction.csv")
        
        X = df[["Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night"]]
        y = df["Addicted_Score"]
        
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Visualización
        plt.figure(figsize=(10, 6))
        plt.scatter(X.iloc[:, 0], y, color='blue', alpha=0.5, label='Real')
        plt.scatter(X.iloc[:, 0], y_pred, color='red', alpha=0.5, label='Predicho')
        plt.title('Relación: Uso diario vs Adicción')
        plt.xlabel('Horas de uso diario')
        plt.ylabel('Puntuación de adicción')
        plt.legend()
        
        if debug:
            print("=== Debug Mode ===")
            print("Primeras 5 predicciones:")
            print(pd.DataFrame({"Real": y[:5], "Predicho": y_pred[:5]}))
            print(f"\nR² Score: {r2_score(y, y_pred):.2f}")
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=120)
        plt.close()
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return None
    

def Predicting_the_level_of_social_media_addiction_2(debug=False):
    try:
        df = pd.read_csv("../../data/Students Social Media Addiction.csv")
        
        X = df[["Age", "Mental_Health_Score"]]
        y = df["Addicted_Score"]
        
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Visualización
        plt.figure(figsize=(10, 6))
        plt.scatter(X.iloc[:, 0], y, color='blue', alpha=0.5, label='Real')
        plt.scatter(X.iloc[:, 0], y_pred, color='red', alpha=0.5, label='Predicho')
        plt.title('Relación: Edad vs Adicción')
        plt.xlabel('Edad')
        plt.ylabel('Puntuación de adicción')
        plt.legend()
        
        if debug:
            print("=== Debug Mode ===")
            print("Primeras 5 predicciones:")
            print(pd.DataFrame({"Real": y[:5], "Predicho": y_pred[:5]}))
            print(f"\nR² Score: {r2_score(y, y_pred):.2f}")
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=120)
        plt.close()
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def Predicting_the_level_of_social_media_addiction_3(debug=False):
    try:
        df = pd.read_csv("../../data/Students Social Media Addiction.csv")
        
        X = df[["Conflicts_Over_Social_Media", "Mental_Health_Score"]]
        y = df["Addicted_Score"]
        
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Visualización
        plt.figure(figsize=(10, 6))
        plt.scatter(X.iloc[:, 0], y, color='blue', alpha=0.5, label='Real')
        plt.scatter(X.iloc[:, 0], y_pred, color='red', alpha=0.5, label='Predicho')
        plt.title('Relación: Conflictos vs Adicción')
        plt.xlabel('Conflictos sobre redes sociales')
        plt.ylabel('Puntuación de adicción')
        plt.legend()
        
        if debug:
            print("=== Debug Mode ===")
            print("Primeras 5 predicciones:")
            print(pd.DataFrame({"Real": y[:5], "Predicho": y_pred[:5]}))
            print(f"\nR² Score: {r2_score(y, y_pred):.2f}")
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=120)
        plt.close()
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return None
    