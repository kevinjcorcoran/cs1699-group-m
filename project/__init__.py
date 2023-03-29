from flask import Flask
import os

# Primary Directories
basePath = os.path.abspath(os.path.dirname(__file__))
template_dir = os.path.join(basePath, 'templates')

def create_app():
    app = Flask(__name__, template_folder=template_dir)

    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app
