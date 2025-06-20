from .logistic_regression import train_and_save as train_lr
from .random_forest import train_and_save as train_rf
from .knn import train_and_save as train_knn

def train_all_models():
    print("Entrenando modelo de Regresión Logística...")
    train_lr()
    
    print("\nEntrenando modelo de Random Forest...")
    train_rf()
    
    print("\nEntrenando modelo KNN...")
    train_knn()
    
    print("\n✅ Todos los modelos entrenados exitosamente")

if __name__ == "__main__":
    train_all_models()