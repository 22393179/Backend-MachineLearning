from flask import Blueprint, request, jsonify
from models.Lineal_Regressions.regresion_lineal import (
    Predicting_the_level_of_social_media_addiction,
    Predicting_the_level_of_social_media_addiction_2,
    Predicting_the_level_of_social_media_addiction_3
)

lineal_blueprint = Blueprint('lineal', __name__)

@lineal_blueprint.route('/predict-addiction-1', methods=['GET'])
def predict_addiction_1():
    return _handle_prediction(Predicting_the_level_of_social_media_addiction)

@lineal_blueprint.route('/predict-addiction-2', methods=['GET'])
def predict_addiction_2():
    return _handle_prediction(Predicting_the_level_of_social_media_addiction_2)

@lineal_blueprint.route('/predict-addiction-3', methods=['GET'])
def predict_addiction_3():
    return _handle_prediction(Predicting_the_level_of_social_media_addiction_3)

# Función auxiliar para evitar repetir código
def _handle_prediction(model_function):
    try:
        debug = request.args.get('debug', 'false').lower() == 'true'
        image_base64 = model_function(debug=debug)
        if image_base64 is None:
            return jsonify({"error": "No se pudo generar la imagen"}), 500
        return jsonify({"image": image_base64}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
