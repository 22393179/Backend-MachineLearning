from flask import Flask
from flask_cors import CORS
from controllers.linealControllers import lineal_blueprint
from controllers.logicalControllers import logical_blueprint
from controllers.treeControllers import tree_blueprint
from controllers.clusterControllers import cluster_blueprint
from controllers.predictionControllers import prediction_blueprint

app = Flask(__name__)

# 🔧 Esta es la configuración CORRECTA para permitir el frontend en el puerto 5173
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# Si usas credenciales (cookies, auth headers), añade:
# CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}}, supports_credentials=True)

# Registrar blueprints
app.register_blueprint(lineal_blueprint)
app.register_blueprint(logical_blueprint)
app.register_blueprint(tree_blueprint)
app.register_blueprint(cluster_blueprint)
app.register_blueprint(prediction_blueprint)

@app.route('/')
def index():
    return "La API esta funcionando correctamente!"

if __name__ == '__main__':
    app.run(debug=True, port=8000)
