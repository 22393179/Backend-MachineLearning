from flask import Blueprint, jsonify
from models.Trees.trees import (
    tree_numeric_binary,
    tree_categorical_instagram,
    tree_mixed_conflict
)

tree_blueprint = Blueprint('tree', __name__)

# Función auxiliar para manejar respuestas
def handle_tree_response(func, error_message):
    try:
        image_base64 = func()
        if image_base64 is None:
            return jsonify({"error": error_message}), 500
        return jsonify({"image": image_base64}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@tree_blueprint.route('/tree/numeric-binary', methods=['GET'])
def get_tree_numeric_binary():
    return handle_tree_response(tree_numeric_binary, "No se pudo generar el árbol numérico binario")

@tree_blueprint.route('/tree/categorical-instagram', methods=['GET'])
def get_tree_categorical_instagram():
    return handle_tree_response(tree_categorical_instagram, "No se pudo generar el árbol categórico de Instagram")

@tree_blueprint.route('/tree/mixed-conflict', methods=['GET'])
def get_tree_mixed_conflict():
    return handle_tree_response(tree_mixed_conflict, "No se pudo generar el árbol mixto de conflictos")
