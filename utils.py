import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import os
from sklearn.preprocessing import LabelEncoder
import joblib

# Mapeo de valores categóricos en español a inglés para que coincidan con el CSV
CATEGORICAL_MAPPINGS = {
    'Gender': {
        'Hombre': 'Hombre',
        'Mujer': 'Mujer',
        'Otro': 'Otro'
    },
    'Country': {
        'México': 'Mexico',
        'Brasil': 'Brasil',
        'Perú': 'Peru'
    },
    'Academic_Level': {
        'Primaria': 'Primary',
        'Secundaria': 'Secondary',
        'Preparatoria': 'Preparatory',
        'Técnico Superior Universitario': 'University Higher Technician',
        'Ingeniería': 'Engineer',
        'Posgrado': 'graduate'
    },
    'Most_Used_Platform': {
        'YouTube': 'YouTube',
        'Instagram': 'Instagram',
        'TikTok': 'TikTok',
        'Facebook': 'Facebook',
        'Discord': 'Discord',
        'WhatsApp': 'WhatsApp',
        'Twitter': 'Twitter',
        'Otra': 'Otra'
    },
    'Relationship_Status': {
        'Soltero/a': 'Single',
        'En una relación': 'In a relationship',
        'Casado/a': 'Married',
        'Complicado': 'Complicated'
    }
}

# Columnas categóricas que necesitan codificación
CATEGORICAL_COLUMNS = ['Gender', 'Country', 'Academic_Level', 'Most_Used_Platform', 'Relationship_Status']

def load_data():
    try:
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'Students_Social_Media_Addiction.csv')
        df = pd.read_csv(data_path, encoding='utf-8-sig')
        print("Archivo cargado con éxito usando codificación: utf-8-sig")
        return df
    except Exception as e:
        print(f"Error al cargar el archivo CSV: {e}")
        raise

def preprocess_data(data, categorical_columns=CATEGORICAL_COLUMNS, df=None):
    try:
        # Convertir el diccionario de entrada o DataFrame en un DataFrame
        if isinstance(data, dict):
            input_df = pd.DataFrame([data])
        else:
            input_df = data.copy()

        # Mapear valores categóricos del español al inglés
        for column in categorical_columns:
            if column in input_df and column in CATEGORICAL_MAPPINGS:
                input_value = input_df[column].iloc[0]
                print(f"Valor de entrada para {column}: {input_value}")  # Debugging
                input_df[column] = input_df[column].map(CATEGORICAL_MAPPINGS[column])
                if input_df[column].isna().any():
                    print(f"Advertencia: Valor no válido para {column}: {input_value}, usando valor por defecto 'Secondary'")
                    input_df[column].fillna('Secondary', inplace=True)  # Fallback a un valor válido

        # Si se proporciona un DataFrame de entrenamiento, usar sus codificadores
        encoders = {}
        if df is not None:
            for column in categorical_columns:
                if column in df and column in input_df:
                    le = LabelEncoder()
                    le.fit(df[column].astype(str))
                    encoders[column] = le
                    input_df[column] = le.transform(input_df[column].astype(str))
        else:
            # Para predicción, cargar los encoders desde el DataFrame de entrenamiento
            if categorical_columns:
                df_train = load_data()
                for column in categorical_columns:
                    if column in input_df:
                        le = LabelEncoder()
                        le.fit(df_train[column].astype(str))
                        encoders[column] = le
                        input_df[column] = le.transform(input_df[column].astype(str))

        # Convertir columnas numéricas
        for column in input_df.columns:
            if column not in categorical_columns:
                input_df[column] = pd.to_numeric(input_df[column], errors='coerce')

        # Asegurar que el resultado sea un DataFrame 2D
        if input_df.ndim != 2:
            raise ValueError(f"El DataFrame preprocesado debe ser 2D, pero tiene forma: {input_df.shape}")

        return input_df, encoders
    except Exception as e:
        print(f"Error en preprocess_data: {e}")
        raise

def generate_scatter_plot(df, x_col, y_col, prediction, user_value):
    try:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x=x_col, y=y_col, alpha=0.5)
        plt.scatter(user_value[x_col], prediction, color='red', s=100, label='Predicción')
        plt.title(f'{x_col} vs {y_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.legend()

        # Guardar el gráfico en un buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        return image_base64
    except Exception as e:
        print(f"Error generando el gráfico: {e}")
        return None

def train_and_save_model(model, X, y, model_path):
    try:
        model.fit(X, y)
        joblib.dump(model, model_path)
        print(f"Modelo guardado en {model_path}")
    except Exception as e:
        print(f"Error al entrenar o guardar el modelo: {e}")
        raise

def load_model(model_path, train_function):
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            return model
        else:
            train_function()
            model = joblib.load(model_path)
            return model
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        raise