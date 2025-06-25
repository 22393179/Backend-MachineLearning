from flask import Flask
from flask_cors import CORS
from controllers.linealControllers import lineal_blueprint
from controllers.logicalControllers import logical_blueprint
from controllers.treeControllers import tree_blueprint
from controllers.clusterControllers import cluster_blueprint

app = Flask(__name__)
CORS(app)


app.register_blueprint(lineal_blueprint)
app.register_blueprint(logical_blueprint)
app.register_blueprint(tree_blueprint)
app.register_blueprint(cluster_blueprint)

@app.route('/')
def index():
    return "La API esta funcionando correctamente!"

if __name__ == '__main__':
    app.run(debug=True)