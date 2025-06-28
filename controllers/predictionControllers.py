from flask import Blueprint, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

from models.Lineal_Regressions.regresion_lineal import (
    Predicting_the_level_of_social_media_addiction,
    Predicting_the_level_of_social_media_addiction_2,
    Predicting_the_level_of_social_media_addiction_3
)
from models.Logical_Regresions.logical_regresions import (
    logistic_binary,
    logistic_multiclass
)
from models.Trees.trees import (
    tree_numeric_binary
)

prediction_blueprint = Blueprint('prediction', __name__)

# Enable CORS for this blueprint
CORS(prediction_blueprint, origins=["http://localhost:5173"])

def get_data_path():
    """Helper function to get the correct dataset path"""
    return Path(__file__).parent.parent / "data" / "Students Social Media Addiction.csv"

@prediction_blueprint.route('/predict-user-addiction', methods=['POST'])
def predict_user_addiction():
    try:
        # Get data from request body
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided in the request"}), 400

        # Validate required fields (including Gender)
        required_fields = [
            'Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night',
            'Conflicts_Over_Social_Media', 'Relationship_Status', 
            'Country', 'Academic_Level', 'Most_Used_Platform', 'Gender'
        ]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

        # Validate numeric fields
        try:
            age = int(data['Age'])
            avg_daily_usage = float(data['Avg_Daily_Usage_Hours'])
            sleep_hours = float(data['Sleep_Hours_Per_Night'])
            conflicts = int(data['Conflicts_Over_Social_Media'])
            if age < 0 or avg_daily_usage < 0 or sleep_hours < 0 or conflicts < 0:
                return jsonify({"error": "Numeric fields cannot be negative"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "Numeric fields must be valid (integers or decimals)"}), 400

        # Load dataset and validate
        try:
            df = pd.read_csv(get_data_path())
        except Exception as e:
            logging.error(f"Failed to load dataset: {str(e)}")
            return jsonify({"error": f"Failed to load dataset: {str(e)}"}), 500

        logging.debug(f"Dataset loaded with shape: {df.shape}")
        logging.debug(f"Dataset columns: {df.columns.tolist()}")
        logging.debug(f"First 5 rows:\n{df.head().to_dict()}")
        logging.debug(f"Missing values:\n{df[['Avg_Daily_Usage_Hours', 'Addicted_Score']].isnull().sum().to_dict()}")

        # Check for required columns
        expected_columns = ['Avg_Daily_Usage_Hours', 'Addicted_Score', 'Mental_Health_Score']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"Missing columns in dataset: {missing_columns}")
            return jsonify({"error": f"Missing columns in dataset: {', '.join(missing_columns)}"}), 500

        # Clean dataset: Fill NaN with mean for numeric columns
        df['Avg_Daily_Usage_Hours'] = df['Avg_Daily_Usage_Hours'].fillna(df['Avg_Daily_Usage_Hours'].mean())
        df['Addicted_Score'] = df['Addicted_Score'].fillna(df['Addicted_Score'].mean())
        logging.debug(f"After filling NaN, missing values:\n{df[['Avg_Daily_Usage_Hours', 'Addicted_Score']].isnull().sum().to_dict()}")

        if df['Addicted_Score'].isnull().all() or df['Avg_Daily_Usage_Hours'].isnull().all():
            logging.error("All values in Addicted_Score or Avg_Daily_Usage_Hours are still NaN after filling")
            return jsonify({"error": "Dataset contains no valid data for Addicted_Score or Avg_Daily_Usage_Hours"}), 500

        mean_mental_health_score = df['Mental_Health_Score'].mean()

        # Validate categorical fields
        valid_gender = df['Gender'].unique().tolist() if 'Gender' in df.columns else ['Male', 'Female', 'Other']
        valid_relationship_status = df['Relationship_Status'].unique().tolist()
        valid_academic_level = df['Academic_Level'].unique().tolist()
        valid_most_used_platform = df['Most_Used_Platform'].unique().tolist()
        valid_country = df['Country'].unique().tolist()

        if data['Gender'] not in valid_gender:
            return jsonify({"error": f"Invalid Gender. Valid values: {', '.join(valid_gender)}"}), 400
        if data['Relationship_Status'] not in valid_relationship_status:
            return jsonify({"error": f"Invalid Relationship_Status. Valid values: {', '.join(valid_relationship_status)}"}), 400
        if data['Academic_Level'] not in valid_academic_level:
            return jsonify({"error": f"Invalid Academic_Level. Valid values: {', '.join(valid_academic_level)}"}), 400
        if data['Most_Used_Platform'] not in valid_most_used_platform:
            return jsonify({"error": f"Invalid Most_Used_Platform. Valid values: {', '.join(valid_most_used_platform)}"}), 400
        if data['Country'] not in valid_country:
            return jsonify({"error": f"Invalid Country. Valid values: {', '.join(valid_country)}"}), 400

        # Create DataFrame with user data
        user_data = pd.DataFrame({
            'Age': [data['Age']],
            'Avg_Daily_Usage_Hours': [data['Avg_Daily_Usage_Hours']],
            'Sleep_Hours_Per_Night': [data['Sleep_Hours_Per_Night']],
            'Mental_Health_Score': [mean_mental_health_score],
            'Conflicts_Over_Social_Media': [data['Conflicts_Over_Social_Media']],
            'Relationship_Status': [data['Relationship_Status']],
            'Country': [data['Country']],
            'Academic_Level': [data['Academic_Level']],
            'Most_Used_Platform': [data['Most_Used_Platform']],
            'Gender': [data['Gender']]
        })

        # --- Linear Regression 1 (Usage + Sleep vs Addiction) ---
        X_linear_1 = df[["Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night"]]
        y_linear_1 = df["Addicted_Score"]
        model_linear_1 = LinearRegression()
        model_linear_1.fit(X_linear_1, y_linear_1)
        user_X_linear_1 = user_data[["Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night"]]
        addiction_score_1 = model_linear_1.predict(user_X_linear_1)[0]

        # --- Linear Regression 2 (Age + Mental Health vs Addiction) ---
        X_linear_2 = df[["Age", "Mental_Health_Score"]]
        y_linear_2 = df["Addicted_Score"]
        model_linear_2 = LinearRegression()
        model_linear_2.fit(X_linear_2, y_linear_2)
        user_X_linear_2 = user_data[["Age", "Mental_Health_Score"]]
        addiction_score_2 = model_linear_2.predict(user_X_linear_2)[0]

        # --- Linear Regression 3 (Conflicts + Mental Health vs Addiction) ---
        X_linear_3 = df[["Conflicts_Over_Social_Media", "Mental_Health_Score"]]
        y_linear_3 = df["Addicted_Score"]
        model_linear_3 = LinearRegression()
        model_linear_3.fit(X_linear_3, y_linear_3)
        user_X_linear_3 = user_data[["Conflicts_Over_Social_Media", "Mental_Health_Score"]]
        addiction_score_3 = model_linear_3.predict(user_X_linear_3)[0]

        # --- Logistic Binary (High vs Low Addiction) ---
        df['Addiction_High'] = (df['Addicted_Score'] > 7).astype(int)
        X_logistic_binary = df[['Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night']]
        y_logistic_binary = df['Addiction_High']
        model_logistic_binary = LogisticRegression()
        model_logistic_binary.fit(X_logistic_binary, y_logistic_binary)
        user_X_logistic_binary = user_data[['Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night']]
        addiction_binary = model_logistic_binary.predict(user_X_logistic_binary)[0]
        addiction_binary_prob = model_logistic_binary.predict_proba(user_X_logistic_binary)[0][1]

        # --- Logistic Multiclass (Low, Medium, High) with Gender ---
        df['Addiction_Level'] = pd.cut(df['Addicted_Score'], 
                                      bins=[0, 4, 7, 10], 
                                      labels=['Bajo', 'Medio', 'Alto'])
        df_encoded = pd.get_dummies(df, columns=['Gender'], prefix='Gender')
        user_data_encoded = pd.get_dummies(user_data, columns=['Gender'], prefix='Gender')
        for col in df_encoded.columns:
            if col.startswith('Gender_') and col not in user_data_encoded.columns:
                user_data_encoded[col] = 0
        X_logistic_multi = df_encoded[['Avg_Daily_Usage_Hours', 'Age', 'Mental_Health_Score'] + [col for col in df_encoded.columns if col.startswith('Gender_')]]
        y_logistic_multi = df['Addiction_Level']
        model_logistic_multi = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        model_logistic_multi.fit(X_logistic_multi, y_logistic_multi)
        user_X_logistic_multi = user_data_encoded[['Avg_Daily_Usage_Hours', 'Age', 'Mental_Health_Score'] + [col for col in df_encoded.columns if col.startswith('Gender_')]]
        addiction_multi = model_logistic_multi.predict(user_X_logistic_multi)[0]
        addiction_multi_probs = model_logistic_multi.predict_proba(user_X_logistic_multi)[0]
        multi_probs_dict = dict(zip(model_logistic_multi.classes_, addiction_multi_probs))

        # --- Decision Tree Binary ---
        X_tree_binary = df[['Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night']]
        y_tree_binary = df['Addiction_High']
        model_tree_binary = DecisionTreeClassifier(max_depth=3)
        model_tree_binary.fit(X_tree_binary, y_tree_binary)
        user_X_tree_binary = user_data[['Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night']]
        tree_binary_pred = model_tree_binary.predict(user_X_tree_binary)[0]

        # --- Generate Plots ---
        # Scatter Plot: Avg_Daily_Usage_Hours vs Addicted_Score
        plt.figure(figsize=(10, 6))
        valid_data = df[['Avg_Daily_Usage_Hours', 'Addicted_Score']].dropna()
        if not valid_data.empty:
            plt.scatter(valid_data['Avg_Daily_Usage_Hours'], valid_data['Addicted_Score'], alpha=0.5, label='Dataset', color='blue')
            logging.debug(f"Plotted {len(valid_data)} dataset points for scatter plot")
        else:
            logging.error("No valid data for scatter plot after dropping NaN")
        plt.scatter(user_data['Avg_Daily_Usage_Hours'], addiction_score_1, color='red', s=100, label='User', marker='*')
        plt.xlabel('Daily Usage Hours')
        plt.ylabel('Addiction Score')
        plt.title('Daily Usage vs. Addiction Score')
        plt.legend()
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=120)
        plt.close()
        scatter_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Histogram: Addicted_Score distribution
        plt.figure(figsize=(10, 6))
        valid_hist_data = df['Addicted_Score'].dropna()
        if not valid_hist_data.empty:
            plt.hist(valid_hist_data, bins=20, alpha=0.7, color='blue', label='Dataset')
            logging.debug(f"Plotted histogram with {len(valid_hist_data)} values")
        else:
            logging.error("No valid data for histogram")
        plt.axvline(x=addiction_score_1, color='red', linestyle='--', label='User')
        plt.xlabel('Addiction Score')
        plt.ylabel('Frequency')
        plt.title('Addiction Score Distribution')
        plt.legend()
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=120)
        plt.close()
        histogram_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Decision Tree Plot
        plt.figure(figsize=(12, 8))
        plot_tree(model_tree_binary, 
                 feature_names=['Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night'],
                 class_names=['Low', 'High'],
                 filled=True, 
                 rounded=True)
        plt.title('Decision Tree for Binary Addiction Classification')
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=120)
        plt.close()
        tree_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # --- Result Interpretation ---
        addiction_level_interpretation = {
            'Bajo': 'The person shows a low level of social media addiction.',
            'Medio': 'The person has a moderate level of social media addiction, which may require attention.',
            'Alto': 'The person shows a high level of social media addiction, intervention is recommended.'
        }
        binary_interpretation = {
            0: 'Low or moderate addiction.',
            1: 'High addiction, further evaluation recommended.'
        }

        # Response JSON
        response = {
            "status": "success",
            "predictions": {
                "linear_regression": {
                    "model_1_usage_sleep": round(float(addiction_score_1), 2),
                    "model_2_age_mental": round(float(addiction_score_2), 2),
                    "model_3_conflicts_mental": round(float(addiction_score_3), 2),
                    "note": "Models 2 and 3 use an imputed average for Mental_Health_Score"
                },
                "logistic_binary": {
                    "prediction": int(addiction_binary),
                    "probability_high": round(float(addiction_binary_prob), 2),
                    "interpretation": binary_interpretation[int(addiction_binary)]
                },
                "logistic_multiclass": {
                    "prediction": str(addiction_multi),
                    "probabilities": {k: round(float(v), 2) for k, v in multi_probs_dict.items()},
                    "interpretation": addiction_level_interpretation[str(addiction_multi)],
                    "note": "This model uses an imputed average for Mental_Health_Score and includes Gender"
                },
                "decision_tree_binary": {
                    "prediction": int(tree_binary_pred),
                    "interpretation": binary_interpretation[int(tree_binary_pred)]
                }
            },
            "plots": {
                "scatter_plot": scatter_plot,
                "histogram_plot": histogram_plot,
                "tree_plot": tree_plot
            }
        }
        return jsonify(response), 200

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"error": f"Error in prediction: {str(e)}"}), 500