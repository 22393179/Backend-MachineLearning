from flask import Blueprint, request, jsonify
from flask_cors import CORS
from models.Clusters.cluster import Addict_age_to_daily_hours_relation

cluster_blueprint = Blueprint('cluster', __name__)

# ✅ Habilita CORS solo para este blueprint
CORS(cluster_blueprint, origins=["http://localhost:5173"])

@cluster_blueprint.route('/addict-age-daily-hours', methods=['GET'])
def addict_age_daily_hours():
    result = Addict_age_to_daily_hours_relation()

    if not result["success"]:
        return jsonify({
            "error": result["message"] or "Error interno en el servidor"
        }), 500

    return jsonify({
        "plot_base64": result["plot"]
    }), 200
