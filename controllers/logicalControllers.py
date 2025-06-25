from flask import Blueprint, request, jsonify
from models.Logical_Regresions.logical_regresions import (
    logistic_binary,
    logistic_multiclass,
    logistic_regularized,
    logistic_platform
)

logical_blueprint = Blueprint('logical', __name__)

# Función reutilizable para manejar errores
def handle_logistic_response(func):
    try:
        result = func()
        if result is None:
            return jsonify({'status': 'error', 'message': 'No se generó la imagen'}), 500
        return jsonify({'status': 'success', 'data': result}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@logical_blueprint.route('/logistic/binary', methods=['GET'])
def get_logistic_binary():
    return handle_logistic_response(logistic_binary)

@logical_blueprint.route('/logistic/multiclass', methods=['GET'])
def get_logistic_multiclass():
    return handle_logistic_response(logistic_multiclass)

@logical_blueprint.route('/logistic/regularized', methods=['GET'])
def get_logistic_regularized():
    return handle_logistic_response(logistic_regularized)

@logical_blueprint.route('/logistic/platform', methods=['GET'])
def get_logistic_platform():
    return handle_logistic_response(logistic_platform)
