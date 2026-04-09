import os
from flask import Flask
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect
from flask_bcrypt import Bcrypt
from flask_talisman import Talisman
import firebase_admin
from firebase_admin import credentials, firestore

csrf = CSRFProtect()
bcrypt = Bcrypt()
login_manager = LoginManager()
login_manager.login_view = 'routes.login'
login_manager.login_message = ""
login_manager.login_message_category = 'info'

def create_app(secret_key):
    app = Flask(__name__)
    app.config['SECRET_KEY'] = secret_key

    from datetime import timedelta
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
    app.config['SESSION_REFRESH_EACH_REQUEST'] = True
    app.config['UPLOAD_FOLDER'] = os.path.join(app.instance_path, 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    app.config['MODELS'] = {}
    app.config['WTF_CSRF_ENABLED'] = False

    # ── Security headers via Talisman ──
    csp = {
        'default-src': ["'self'"],
        'script-src': [
            "'self'",
            "'unsafe-inline'",          # needed for inline <script> blocks
            "https://www.gstatic.com",  # Firebase JS SDK
            "https://code.jquery.com",
            "https://cdn.datatables.net",
            "https://cdn.jsdelivr.net",
            "https://cdnjs.cloudflare.com",
            "https://fonts.googleapis.com",
        ],
        'style-src': [
            "'self'",
            "'unsafe-inline'",          # needed for inline styles
            "https://fonts.googleapis.com",
            "https://cdn.datatables.net",
            "https://cdnjs.cloudflare.com",
        ],
        'font-src': [
            "'self'",
            "https://fonts.gstatic.com",
        ],
        'img-src': [
            "'self'",
            "data:",                    # base64 preview images
        ],
        'connect-src': [
            "'self'",
            "https://*.googleapis.com", # Firestore/Firebase API calls
            "https://*.firebaseio.com",
        ],
        'frame-src': ["'none'"],
        'object-src': ["'none'"],
    }

    Talisman(
        app,
        force_https=False,              # set True on Railway/Render (they handle HTTPS)
        strict_transport_security=False, # same — let the host handle HSTS
        session_cookie_secure=False,    # set True in production
        session_cookie_http_only=True,  # prevent JS access to session cookie
        content_security_policy=csp,
        referrer_policy='strict-origin-when-cross-origin',
        feature_policy={
            'geolocation': "'none'",
            'camera': "'none'",
            'microphone': "'none'",
        }
    )

    bcrypt.init_app(app)
    login_manager.init_app(app)

    @app.context_processor
    def inject_notifications():
        from flask_login import current_user
        from flask import request as flask_request
        notifications = []
        unread_count  = 0
        try:
            if current_user.is_authenticated and current_user.role == 'patient':
                notifs_snap = db.collection("notifications") \
                    .where("user_id", "==", current_user.id) \
                    .order_by("created_at", direction=firestore.Query.DESCENDING) \
                    .limit(20).get()
                for n in notifs_snap:
                    obj = n.to_dict()
                    obj["id"] = n.id
                    notifications.append(obj)
                unread_count = sum(1 for n in notifications if not n.get("read"))
        except Exception:
            pass
        return dict(notifications=notifications, unread_count=unread_count)

    @app.before_request
    def check_csrf():
        return None

    if not firebase_admin._apps:
        cred = credentials.Certificate('firebase-key.json')
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'brainstrokedetection.firebasestorage.app'
        })
        print("Firebase Admin SDK initialized successfully.")

    db = firestore.client()

    from .routes import limiter
    limiter.init_app(app)

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