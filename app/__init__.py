import os
from flask import Flask
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect
from flask_bcrypt import Bcrypt
import firebase_admin
from firebase_admin import credentials, firestore

csrf = CSRFProtect()
bcrypt = Bcrypt()
login_manager = LoginManager()
login_manager.login_view = 'routes.login'
login_manager.login_message = ""          # suppress default "Please log in" flash
login_manager.login_message_category = 'info'

def create_app(secret_key):
    app = Flask(__name__)
    app.config['SECRET_KEY'] = secret_key

    # Session timeout — auto logout after 30 minutes of inactivity
    from datetime import timedelta
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
    app.config['SESSION_REFRESH_EACH_REQUEST'] = True

    app.config['UPLOAD_FOLDER'] = os.path.join(app.instance_path, 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    app.config['MODELS'] = {}

    app.config['WTF_CSRF_ENABLED'] = False
    bcrypt.init_app(app)
    login_manager.init_app(app)

    @app.before_request
    def check_csrf():
        return None

    # Initialize Firebase only once
    if not firebase_admin._apps:
        cred = credentials.Certificate('firebase-key.json')
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'brainstrokedetection.firebasestorage.app'
        })
        print("Firebase Admin SDK initialized successfully.")

    # Firestore client
    db = firestore.client()

    # Rate Limiter
    from .routes import limiter
    limiter.init_app(app)

    # Blueprints
    from . import models, routes
    app.register_blueprint(routes.bp)

    @login_manager.user_loader
    def load_user(user_id):
        return models.User.get_by_id(user_id)

    from .ml import model_loader
    app.config['MODELS']['stroke'] = model_loader
    print("ML model loaded and ready.")

    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    return app