import os
from functools import wraps
from flask import Blueprint, render_template, request, redirect, url_for, current_app, flash, jsonify, abort
from flask_login import login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
from .forms import UploadForm, LoginForm, RegistrationForm
from .models import User
from .ml.model_loader import predict_stroke_risk
from firebase_admin import auth, storage, firestore, exceptions

bp = Blueprint('routes', __name__)
db = firestore.client()

# -------------------------------
# Utility
# -------------------------------
def log_event(user_id, role, action, details=None):
    db.collection("logs").add({
        "user_id": user_id,
        "role": role,
        "action": action,
        "details": details or {},
        "created_at": firestore.SERVER_TIMESTAMP
    })

def role_required(*roles):
    def decorator(f):
        @wraps(f)
        @login_required
        def wrapped(*args, **kwargs):
            if not current_user.is_authenticated:
                return redirect(url_for('routes.login'))
            if roles and (current_user.role not in roles):
                flash("Unauthorized access.", "danger")
                return redirect(url_for('routes.home'))
            return f(*args, **kwargs)
        return wrapped
    return decorator

# -----------------------------
# Auth pages
# -----------------------------
@bp.route('/login', methods=['GET'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('routes.home'))
    return render_template('login.html',
        api_key=os.environ.get('FIREBASE_API_KEY'),
        auth_domain=os.environ.get('FIREBASE_AUTH_DOMAIN'),
        project_id=os.environ.get('FIREBASE_PROJECT_ID'))

@bp.route('/register', methods=['GET'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('routes.home'))
    return render_template('register.html',
        api_key=os.environ.get('FIREBASE_API_KEY'),
        auth_domain=os.environ.get('FIREBASE_AUTH_DOMAIN'),
        project_id=os.environ.get('FIREBASE_PROJECT_ID'))

# -----------------------------
# Auth endpoints
# -----------------------------
@bp.route('/session_login', methods=['POST'])
def session_login():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON.'}), 400
    data = request.get_json(silent=True)
    id_token = data.get('idToken')
    if not id_token:
        return jsonify({'error': 'ID token not provided.'}), 400
    try:
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token['uid']
        user = User.get_by_id(uid)
        if not user:
            return jsonify({'error': 'User not found in Firestore.'}), 404
        login_user(user)
        log_event(uid, user.role, "login")
        return jsonify({'success': True, 'message': 'Login successful'}), 200
    except exceptions.InvalidIdTokenError:
        return jsonify({'error': 'Invalid ID token.'}), 401
    except Exception as e:
        return jsonify({'error': str(e)}), 500

BLOCKED_EMAIL_DOMAINS = {
    'example.com', 'example.net', 'example.org',
    'test.com', 'test.net', 'fake.com',
    'mailinator.com', 'guerrillamail.com', 'tempmail.com',
    'throwaway.email', 'yopmail.com', 'sharklasers.com',
    'trashmail.com', 'dispostable.com'
}

@bp.route('/session_register', methods=['POST'])
def session_register():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON.'}), 400
    data = request.get_json(silent=True)
    id_token = data.get('idToken')
    role = data.get('role', 'patient')
    if not id_token:
        return jsonify({'error': 'ID token not provided.'}), 400
    try:
        decoded_token = auth.verify_id_token(id_token)
        uid = decoded_token['uid']
        email = decoded_token.get('email', '')
 
        # Backend domain validation (defence in depth)
        domain = email.split('@')[-1].lower() if '@' in email else ''
        if domain in BLOCKED_EMAIL_DOMAINS:
            return jsonify({'error': 'Please use a real email address, not a test or example domain.'}), 400
 
        user_ref = db.collection("users").document(uid)
        if user_ref.get().exists:
            return jsonify({'error': 'User already registered.'}), 400
        if role not in ["patient", "doctor"]:
            role = "patient"
        user_ref.set({
            "email": email,
            "role": role,
            "created_at": firestore.SERVER_TIMESTAMP
        })
        log_event(uid, role, "register", {"email": email})
        return jsonify({'success': True, 'uid': uid, 'role': role}), 200
    except exceptions.InvalidIdTokenError:
        return jsonify({'error': 'Invalid ID token.'}), 401
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('routes.login'))


# -------------------------------
# General Pages
# -------------------------------
@bp.route('/', methods=['GET', 'POST'])
@bp.route('/home', methods=['GET', 'POST'])
@login_required
def home():
    return render_template('index.html')

@bp.route('/about')
def about():
    return render_template('about.html')

@bp.route('/comparison')
def comparison():
    return render_template('comparison.html')

@bp.route('/get_csrf_token', methods=['GET'])
def get_csrf_token():
    from flask_wtf.csrf import generate_csrf
    return jsonify({'csrf_token': generate_csrf()})


# -------------------------------
# Prediction
# -------------------------------
@bp.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    form = UploadForm()
    result = None
    selected_model = "resnet50"

    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        allowed_extensions = {'jpg', 'jpeg', 'png'}

        if '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions:
            upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(upload_path)
                selected_model = request.form.get("model", "resnet50")
                result = predict_stroke_risk(upload_path, model_name=selected_model)

                if result:
                    pred_ref = db.collection("predictions").add({
                        "user_id": current_user.id,
                        "doctor_ids": [],
                        "file_path": filename,
                        "model_used": selected_model,
                        "result": result["predicted_class"],
                        "probabilities": result["probabilities"],
                        "explainability_paths": {
                            "gradcam": result.get("gradcam_path"),
                            "original": result.get("original_path")
                        },
                        "status": "pending_review",
                        "created_at": firestore.SERVER_TIMESTAMP
                    })
                    log_event(current_user.id, current_user.role, "predict", {
                        "model": selected_model,
                        "prediction_id": pred_ref[1].id if isinstance(pred_ref, tuple) else None
                    })
                    flash("Prediction saved. You can view it in your dashboard.", "success")

            except Exception as e:
                flash(f"Prediction failed: {str(e)}", "danger")
            finally:
                if os.path.exists(upload_path):
                    os.remove(upload_path)
        else:
            flash('Invalid file type. Please upload a .jpg or .png image.', 'warning')

    return render_template('predict.html', form=form, result=result, selected_model=selected_model)


# -----------------------------
# Dashboards
# -----------------------------
@bp.route("/dashboard/admin")
@role_required("admin")
def dashboard_admin():
    users_snap = db.collection("users").get()
    users = []
    for d in users_snap:
        obj = d.to_dict()
        obj["id"] = d.id
        users.append(obj)

    try:
        logs_snap = db.collection("logs") \
            .order_by("created_at", direction=firestore.Query.DESCENDING) \
            .limit(500).get()
        logs = []
        for d in logs_snap:
            obj = d.to_dict()
            obj["id"] = d.id
            logs.append(obj)
    except Exception as e:
        print(f"[WARN] Firestore log ordering failed, sorting in Python: {e}")
        logs_snap = db.collection("logs").limit(500).get()
        logs = []
        for d in logs_snap:
            obj = d.to_dict()
            obj["id"] = d.id
            logs.append(obj)
        logs.sort(key=lambda x: x.get("created_at") or 0, reverse=True)

    return render_template('dashboard_admin.html', users=users, logs=logs)


@bp.route("/dashboard/doctor")
@role_required("doctor")
def dashboard_doctor():
    try:
        cases_snap = db.collection("predictions") \
            .where("status", "==", "pending_review") \
            .order_by("created_at", direction=firestore.Query.DESCENDING) \
            .limit(200).get()
    except Exception as e:
        print(f"[WARN] Doctor dashboard ordering failed: {e}")
        cases_snap = db.collection("predictions") \
            .where("status", "==", "pending_review") \
            .limit(200).get()

    cases = []
    for d in cases_snap:
        obj = d.to_dict()
        obj["id"] = d.id

        # Attach patient email
        user_ref = db.collection("users").document(obj["user_id"]).get()
        obj["patient_email"] = user_ref.to_dict().get("email") if user_ref.exists else obj["user_id"]

        # Build result_display with confidence
        if "probabilities" in obj and isinstance(obj["probabilities"], dict):
            max_class = max(obj["probabilities"], key=obj["probabilities"].get)
            confidence = round(obj["probabilities"][max_class] * 100, 2)
            obj["result_display"] = f"{obj['result']} ({confidence}%)"
        else:
            obj["result_display"] = obj.get("result", "N/A")

        # ISSUE 5 FIX: ensure explainability_paths is always present
        # (older predictions may not have this field)
        if "explainability_paths" not in obj:
            obj["explainability_paths"] = {"gradcam": None, "original": None}

        cases.append(obj)

    return render_template('dashboard_doctor.html', cases=cases)


@bp.route("/dashboard/patient")
@role_required("patient")
def dashboard_patient():
    try:
        preds_snap = db.collection("predictions") \
            .where("user_id", "==", current_user.id) \
            .order_by("created_at", direction=firestore.Query.DESCENDING) \
            .limit(200).get()
    except Exception as e:
        print(f"[WARN] Patient dashboard ordering failed: {e}")
        preds_snap = db.collection("predictions") \
            .where("user_id", "==", current_user.id) \
            .limit(200).get()

    predictions = []
    for d in preds_snap:
        obj = d.to_dict()
        obj["id"] = d.id

        # Always fetch reviews so the clot count logic works correctly
        reviews_snap = db.collection("reviews") \
            .where("prediction_id", "==", obj["id"]).get()
        reviews = []
        for r in reviews_snap:
            rdict = r.to_dict()
            doc_ref = db.collection("users").document(rdict["doctor_id"]).get()
            rdict["doctor_email"] = doc_ref.to_dict().get("email") if doc_ref.exists else rdict["doctor_id"]
            reviews.append(rdict)
        obj["reviews"] = reviews

        predictions.append(obj)

    return render_template('dashboard_patient.html', predictions=predictions)


# -----------------------------
# Admin: update user role
# -----------------------------
@bp.route('/admin/update_role/<user_id>', methods=['POST'])
@role_required("admin")
def update_role(user_id):
    new_role = request.form.get('role')
    if new_role not in ['patient', 'doctor', 'admin']:
        flash("Invalid role.", "warning")
        return redirect(url_for('routes.dashboard_admin'))
    try:
        db.collection("users").document(user_id).update({"role": new_role})
        log_event(current_user.id, current_user.role, "update_role", {"target_user": user_id, "new_role": new_role})
        flash("Role updated.", "success")
    except Exception as e:
        flash(f"Failed to update role: {str(e)}", "danger")
    return redirect(url_for('routes.dashboard_admin'))


# -----------------------------
# Doctor: submit review
# -----------------------------
@bp.route("/review/<prediction_id>", methods=["POST"])
@role_required("doctor")
def review(prediction_id):
    decision = None
    notes = ""
    if request.is_json:
        data = request.get_json(silent=True) or {}
        decision = data.get("decision")
        notes = data.get("notes", "")
    else:
        decision = request.form.get("decision")
        notes = request.form.get("notes", "")

    if decision not in ["agree", "disagree"]:
        flash("Invalid decision.", "warning")
        return redirect(url_for('routes.dashboard_doctor'))

    existing = db.collection("reviews") \
        .where("prediction_id", "==", prediction_id) \
        .where("doctor_id", "==", current_user.id).get()
    if existing:
        flash("You already reviewed this case.", "info")
        return redirect(url_for('routes.dashboard_doctor'))

    db.collection("reviews").add({
        "prediction_id": prediction_id,
        "doctor_id": current_user.id,
        "decision": decision,
        "notes": notes,
        "created_at": firestore.SERVER_TIMESTAMP
    })

    db.collection("predictions").document(prediction_id).update({"status": "reviewed"})
    log_event(current_user.id, "doctor", "review", {"prediction_id": prediction_id, "decision": decision})
    flash("Review submitted.", "success")
    return redirect(url_for('routes.dashboard_doctor'))


# -----------------------------
# Error handlers
# -----------------------------
@bp.app_errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@bp.app_errorhandler(403)
def forbidden(e):
    return render_template('403.html'), 403