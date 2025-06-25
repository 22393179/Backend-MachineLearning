import matplotlib
matplotlib.use('Agg')
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from io import BytesIO
from pathlib import Path
import base64
import pandas as pd

def get_data_path():
    """Función auxiliar para obtener la ruta correcta del dataset"""
    return Path(__file__).parent.parent.parent / "data" / "Students Social Media Addiction.csv"

def Addict_age_to_daily_hours_relation(debug=False):
    response = {"success": False, "plot": None, "message": None}
    try:
        df = pd.read_csv(get_data_path())

        model = KMeans(n_clusters=3, random_state=42)
        df["cluster"] = model.fit_predict(df[["Avg_Daily_Usage_Hours", "Addicted_Score", "Age"]])

        # 3D plot
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(df["Addicted_Score"], df["Avg_Daily_Usage_Hours"], df["Age"],
                             c=df["cluster"], cmap='viridis', marker='o', alpha=0.6)

        ax.set_xlabel('Puntuación de adicción')
        ax.set_ylabel('Horas de uso diario')
        ax.set_zlabel('Edad')
        plt.title('Relación entre Adicción, Uso Diario y Edad')
        plt.colorbar(scatter, label='Cluster')

        if debug:
            print("Debug Info:", df.head())

        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=120)
        plt.close()

        response["success"] = True
        response["plot"] = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return response
    
    except Exception as e:
        response["message"] = f"Error: {str(e)}"
        return response
