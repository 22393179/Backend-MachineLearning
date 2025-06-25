from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd

def Addict_age_to_daily_hours_relation(debug=False):
    try:
        df = pd.read_csv("../../data/Students Social Media Addiction.csv")

        KMeans_model = KMeans(n_clusters=3, random_state=42)
        df_kmeans = KMeans_model.fit_predict(df[["Avg_Daily_Usage_Hours", "Addicted_Score", "Age"]])

        X = df["Addicted_Score"]
        y = df["Avg_Daily_Usage_Hours"]
        z = df["Age"]

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(X, y, z, c=df_kmeans, cmap='viridis', marker='o', alpha=0.6)
        ax.set_xlabel('Puntuación de adicción')
        ax.set_ylabel('Horas de uso diario')
        ax.set_zlabel('Edad')

        plt.title('Relación entre Puntuación de Adicción, Uso Diario de redes y Edad')
        plt.colorbar(scatter, label='Cluster')
        if debug:
            print("=== Debug Mode ===")
            print("Primeras 5 filas del DataFrame:")
            print(df.head())
            print(f"\nNúmero de clusters encontrados: {len(set(df_kmeans))}")
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=120)
        plt.close()
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return None
