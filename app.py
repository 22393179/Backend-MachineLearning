from flask import Flask, request, jsonify
from flask_cors import CORS
from predict_model_01 import model_01_blueprint
from predict_model_02 import model_02_blueprint
from predict_model_03 import model_03_blueprint
from predict_model_04 import model_04_blueprint
from predict_model_05 import model_05_blueprint
from predict_model_06 import model_06_blueprint
from predict_model_07 import model_07_blueprint
from predict_model_08 import model_08_blueprint
from predict_model_09 import model_09_blueprint
from predict_model_10 import model_10_blueprint
from predict_model_11 import model_11_blueprint
from predict_model_12 import model_12_blueprint
from predict_model_13 import model_13_blueprint

app = Flask(__name__)

# Configurar CORS
CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "http://localhost:3000"]}})

# Registrar blueprints
app.register_blueprint(model_01_blueprint, url_prefix='/api')
app.register_blueprint(model_02_blueprint, url_prefix='/api')
app.register_blueprint(model_03_blueprint, url_prefix='/api')
app.register_blueprint(model_04_blueprint, url_prefix='/api')
app.register_blueprint(model_05_blueprint, url_prefix='/api')
app.register_blueprint(model_06_blueprint, url_prefix='/api')
app.register_blueprint(model_07_blueprint, url_prefix='/api')
app.register_blueprint(model_08_blueprint, url_prefix='/api')
app.register_blueprint(model_09_blueprint, url_prefix='/api')
app.register_blueprint(model_10_blueprint, url_prefix='/api')
app.register_blueprint(model_11_blueprint, url_prefix='/api')
app.register_blueprint(model_12_blueprint, url_prefix='/api')
app.register_blueprint(model_13_blueprint, url_prefix='/api')

@app.route('/')
def index():
    return jsonify({"status": "success", "message": "La API está funcionando correctamente!"})

@app.route('/api/predict-all', methods=['POST'])
def predict_all():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'error': 'No se proporcionaron datos'}), 400

        models = [
            ('predict-model-01', 'Addicted_Score_Prediction', 'addicted_score_prediction_plot'),
            ('predict-model-02', 'Sleep_Hours_Prediction', 'sleep_hours_prediction_plot'),
            ('predict-model-03', 'Conflicts_Prediction', 'conflicts_prediction_plot'),
            ('predict-model-04', 'Academic_Impact_Prediction', 'academic_impact_prediction_plot'),
            ('predict-model-05', 'Mental_Health_Risk_Prediction', 'mental_health_risk_prediction_plot'),
            ('predict-model-06', 'Addiction_Classification', 'addiction_classification_plot'),
            ('predict-model-07', 'Most_Used_Platform', 'most_used_platform_plot'),
            ('predict-model-08', 'Relationship_Status', 'relationship_status_plot'),
            ('predict-model-09', 'Cluster', 'cluster_plot'),
            ('predict-model-10', 'Addiction_vs_Sleep', 'addiction_vs_sleep_plot'),
            ('predict-model-11', 'Social_Media_Conflicts', 'social_media_conflicts_plot'),
            ('predict-model-12', 'Academic_Performance_Risk', 'academic_performance_risk_plot'),
            ('predict-model-13', 'General_Mixed_Model', 'general_mixed_model_plot'),
        ]

        predictions = {}
        plots = {}

        for endpoint, pred_key, plot_key in models:
            try:
                response = app.test_client().post(f'/api/{endpoint}', json=data)
                json_resp = response.get_json()
                
                if response.status_code != 200:
                    return jsonify({
                        'status': 'error',
                        'error': f'Error en {endpoint}: {json_resp.get("error", "Respuesta no válida")}'
                    }), 500
                
                if json_resp.get('status') != 'success':
                    return jsonify({
                        'status': 'error',
                        'error': f'Error en {endpoint}: {json_resp.get("error", "Respuesta no válida")}'
                    }), 500
                
                if pred_key not in json_resp.get('predictions', {}):
                    return jsonify({
                        'status': 'error',
                        'error': f'Clave de predicción {pred_key} no encontrada en la respuesta de {endpoint}'
                    }), 500
                
                predictions[pred_key] = json_resp['predictions'][pred_key]
                plots[plot_key] = json_resp.get('plots', {}).get(plot_key, None)

            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'error': f'Error procesando {endpoint}: {str(e)}'
                }), 500

        return jsonify({
            'status': 'success',
            'predictions': predictions,
            'plots': plots
        }), 200

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': f'Error en el servidor: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)