import os
from functools import wraps
from io import BytesIO
from flask import send_file, Blueprint, render_template, request, redirect, url_for, current_app, flash, jsonify, abort
from flask_login import login_user, login_required, logout_user, current_user
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
from .forms import UploadForm, LoginForm, RegistrationForm
from .models import User
from .ml.model_loader import predict_stroke_risk
from firebase_admin import auth, storage, firestore, exceptions
import anthropic as anthropic_sdk
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

bp = Blueprint('routes', __name__)
db = firestore.client()

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[]
)

ALLOWED_MAGIC_BYTES = {
    b'\xff\xd8\xff': 'jpg',   # JPEG
    b'\x89PNG':      'png',   # PNG
}

def is_valid_image(file_path):
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
        return any(header.startswith(magic) for magic in ALLOWED_MAGIC_BYTES)
    except Exception:
        return False

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

def send_email_notification(to_email, patient_name, decision, notes, model_used):
    """Send email to patient when doctor reviews their scan."""
    mail_email    = os.environ.get('MAIL_EMAIL')
    mail_password = os.environ.get('MAIL_PASSWORD')
 
    if not mail_email or not mail_password:
        print("[WARN] Email not configured — skipping email notification")
        return
 
    try:
        decision_text  = "agrees with" if decision == "agree" else "disagrees with"
        decision_emoji = "✔" if decision == "agree" else "✖"
        subject = f"NeuroScan — Your MRI scan has been reviewed"
 
        html = f"""
        <div style="font-family:Arial,sans-serif;max-width:560px;margin:0 auto;background:#f8fafc;padding:24px;border-radius:12px;">
            <div style="background:linear-gradient(135deg,#0d3f7a,#1558a8);border-radius:10px;padding:24px;text-align:center;margin-bottom:24px;">
                <h1 style="color:#fff;font-size:22px;margin:0;">🧠 NeuroScan</h1>
                <p style="color:rgba(255,255,255,0.75);margin:6px 0 0;font-size:13px;">AI-Powered Stroke Detection</p>
            </div>
 
            <div style="background:#fff;border-radius:10px;padding:24px;border:1px solid #e5e7eb;">
                <h2 style="color:#111827;font-size:17px;margin:0 0 8px;">Your scan has been reviewed</h2>
                <p style="color:#6b7280;font-size:14px;margin:0 0 20px;">
                    Hello {patient_name or to_email}, a doctor has reviewed your MRI scan on NeuroScan.
                </p>
 
                <div style="background:#f0f4ff;border-radius:8px;padding:16px;margin-bottom:20px;border-left:4px solid #1558a8;">
                    <div style="font-size:13px;color:#6b7280;margin-bottom:4px;">Model Used</div>
                    <div style="font-weight:600;color:#111827;font-size:15px;">{(model_used or 'N/A').upper()}</div>
                </div>
 
                <div style="background:{'#f0fdf4' if decision == 'agree' else '#fef2f2'};border-radius:8px;padding:16px;margin-bottom:20px;border-left:4px solid {'#0f7a3e' if decision == 'agree' else '#b91c1c'};">
                    <div style="font-size:13px;color:#6b7280;margin-bottom:4px;">Doctor's Decision</div>
                    <div style="font-weight:700;color:{'#0f7a3e' if decision == 'agree' else '#b91c1c'};font-size:15px;">
                        {decision_emoji} The doctor {decision_text} the AI prediction
                    </div>
                    {f'<div style="margin-top:8px;font-size:13px;color:#374151;font-style:italic;">"{notes}"</div>' if notes else ''}
                </div>
 
                <p style="font-size:12px;color:#9ca3af;margin:0;">
                    ⚠ This is an AI-assisted research tool. Please consult a qualified medical professional for clinical advice.
                </p>
            </div>
 
            <div style="text-align:center;margin-top:16px;">
                <p style="font-size:11px;color:#9ca3af;">NeuroScan FYP · Universiti Tunku Abdul Rahman</p>
            </div>
        </div>
        """
 
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From']    = f"NeuroScan <{mail_email}>"
        msg['To']      = to_email
        msg.attach(MIMEText(html, 'html'))
 
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(mail_email, mail_password)
            server.sendmail(mail_email, to_email, msg.as_string())
 
        print(f"[INFO] Email notification sent to {to_email}")
 
    except Exception as e:
        print(f"[WARN] Email notification failed: {e}")

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
        from flask import session
        session.permanent = True
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

# ── ADD these two routes to routes.py ──
# Place them after the logout route, before the General Pages section

@bp.route('/settings')
@login_required
def settings():
    # Fetch extra info from Firestore
    user_doc = db.collection("users").document(current_user.id).get()
    user_data = user_doc.to_dict() if user_doc.exists else {}

    # Format joined date
    joined_at = None
    if user_data.get("created_at"):
        try:
            joined_at = user_data["created_at"].strftime("%d %b %Y")
        except Exception:
            joined_at = str(user_data["created_at"])

    # Count predictions for this user
    try:
        preds = db.collection("predictions") \
            .where("user_id", "==", current_user.id).get()
        prediction_count = len(preds)
    except Exception:
        prediction_count = 0

    # Detect if Google user (no password provider)
    is_google_user = False
    try:
        firebase_user = auth.get_user(current_user.id)
        providers = [p.provider_id for p in firebase_user.provider_data]
        is_google_user = 'google.com' in providers and 'password' not in providers
    except Exception:
        pass

    return render_template('settings.html',
        joined_at=joined_at,
        prediction_count=prediction_count,
        is_google_user=is_google_user,
        api_key=os.environ.get('FIREBASE_API_KEY'),
        auth_domain=os.environ.get('FIREBASE_AUTH_DOMAIN'),
        project_id=os.environ.get('FIREBASE_PROJECT_ID')
    )


@bp.route('/settings/update_name', methods=['POST'])
@login_required
def update_display_name():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON.'}), 400
    data = request.get_json(silent=True)
    name = (data.get('name') or '').strip()
    if not name or len(name) < 2:
        return jsonify({'error': 'Name must be at least 2 characters.'}), 400
    if len(name) > 60:
        return jsonify({'error': 'Name must be under 60 characters.'}), 400
    try:
        db.collection("users").document(current_user.id).update({"name": name})
        current_user.name = name   # update in-session object too
        log_event(current_user.id, current_user.role, "update_name", {"name": name})
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/chat', methods=['POST'])
@login_required
@limiter.limit("20 per minute")
def chat():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON.'}), 400
 
    data = request.get_json(silent=True)
    user_message = (data.get('message') or '').strip()[:1000]
    history = data.get('history', [])
 
    if not user_message:
        return jsonify({'error': 'Message is empty.'}), 400
 
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        return jsonify({'error': 'AI service not configured.'}), 500
 
    try:
        client = anthropic_sdk.Anthropic(api_key=api_key)
 
        # Build message history (last 10 turns max)
        messages = []
        for h in history[-10:]:
            if h.get('role') in ('user', 'assistant') and h.get('content'):
                messages.append({'role': h['role'], 'content': h['content']})
 
        # Ensure last message is from user
        if not messages or messages[-1]['role'] != 'user':
            messages.append({'role': 'user', 'content': user_message})
 
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",  # Fast and cost-effective
            max_tokens=512,
            system="""You are NeuroScan AI, a helpful medical AI assistant integrated into the NeuroScan stroke detection system.
 
Your role is to help users understand:
- Ischemic stroke: what it is, causes, symptoms, and treatment
- How the NeuroScan system works (MRI upload, AI prediction, Grad-CAM, Attention Rollout)
- The deep learning models used (ResNet50, ResNet101, DenseNet121, DenseNet169, EfficientNet-B3, Vision Transformer, Ensemble)
- What Grad-CAM and Attention Rollout heatmaps mean
- How to interpret confidence scores and predictions
- General questions about brain health and stroke prevention
 
Important guidelines:
- Always remind users that this system is a research prototype and NOT a substitute for professional medical diagnosis
- Be concise, friendly, and clear
- Do not diagnose specific patients or interpret specific scan results
- If asked about something outside your scope, politely redirect to relevant topics
- Keep responses under 150 words unless the question requires more detail""",
            messages=messages
        )
 
        reply = response.content[0].text
        return jsonify({'reply': reply}), 200
 
    except anthropic_sdk.APIError as e:
        print(f"[WARN] Anthropic API error: {e}")
        return jsonify({'error': 'AI service temporarily unavailable.'}), 503
    except Exception as e:
        print(f"[WARN] Chat error: {e}")
        return jsonify({'error': str(e)}), 500

# -------------------------------
# General Pages
# -------------------------------
@bp.route('/', methods=['GET', 'POST'])
@bp.route('/home', methods=['GET', 'POST'])
@login_required
def home():
    notifications = []
    unread_count  = 0
    if current_user.role == 'patient':
        try:
            notifs_snap = db.collection("notifications") \
                .where("user_id", "==", current_user.id) \
                .order_by("created_at", direction=firestore.Query.DESCENDING) \
                .limit(20).get()
            for n in notifs_snap:
                obj = n.to_dict()
                obj["id"] = n.id
                notifications.append(obj)
            unread_count = sum(1 for n in notifications if not n.get("read"))
        except Exception as e:
            print(f"[WARN] Notifications fetch failed: {e}")
    return render_template('index.html',
        notifications=notifications,
        unread_count=unread_count)

@bp.route('/about')
def about():
    return render_template('about.html')

@bp.route('/comparison')
def comparison():
    return render_template('comparison.html')


# -------------------------------
# Prediction
# -------------------------------
@bp.route('/predict', methods=['GET', 'POST'])
@login_required
@limiter.limit("10 per minute")
def predict():
    form = UploadForm()
    result = None
    max_prob = None
    selected_model = "resnet50"

    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        allowed_extensions = {'jpg', 'jpeg', 'png'}

        if '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions:
            upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(upload_path)

                # Magic byte validation
                if not is_valid_image(upload_path):
                    os.remove(upload_path)
                    flash('Invalid file. Please upload a real JPEG or PNG image.', 'warning')
                    return render_template('predict.html', form=form, result=result, selected_model=selected_model, max_prob=max_prob)

                selected_model = request.form.get("model", "resnet50")
                result = predict_stroke_risk(upload_path, model_name=selected_model)
                max_prob = max(result["probabilities"].values()) if result else None

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

    return render_template('predict.html', form=form, result=result, selected_model=selected_model, max_prob=max_prob)


# -----------------------------
# Dashboards
# -----------------------------
@bp.route("/dashboard/admin")
@role_required("admin")
def dashboard_admin():
    # Fetch users — build a cache dict {uid: user_data}
    users_snap = db.collection("users").get()
    users = []
    user_cache = {}
    for d in users_snap:
        obj = d.to_dict()
        obj["id"] = d.id
        users.append(obj)
        user_cache[d.id] = obj

    # Fetch logs
    try:
        logs_snap = db.collection("logs") \
            .order_by("created_at", direction=firestore.Query.DESCENDING) \
            .limit(500).get()
        logs = [dict(d.to_dict(), id=d.id) for d in logs_snap]
    except Exception as e:
        print(f"[WARN] Firestore log ordering failed: {e}")
        logs_snap = db.collection("logs").limit(500).get()
        logs = [dict(d.to_dict(), id=d.id) for d in logs_snap]
        logs.sort(key=lambda x: x.get("created_at") or 0, reverse=True)

    # Fetch predictions — use user_cache instead of per-prediction DB call
    try:
        preds_snap = db.collection("predictions") \
            .order_by("created_at", direction=firestore.Query.DESCENDING) \
            .limit(500).get()
    except Exception:
        preds_snap = db.collection("predictions").limit(500).get()

    predictions = []
    for d in preds_snap:
        obj = d.to_dict()
        obj["id"] = d.id
        # Use cache — zero extra DB calls
        user = user_cache.get(obj.get("user_id"), {})
        obj["patient_email"] = user.get("email", obj.get("user_id", "Unknown"))
        predictions.append(obj)

    return render_template('dashboard_admin.html',
        users=users,
        logs=logs,
        predictions=predictions,
        total_predictions=len(predictions)
    )


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

    # Collect all unique user_ids first
    raw_cases = []
    user_ids = set()
    for d in cases_snap:
        obj = d.to_dict()
        obj["id"] = d.id
        raw_cases.append(obj)
        if obj.get("user_id"):
            user_ids.add(obj["user_id"])

    # Batch fetch all users in one go
    user_cache = {}
    for uid in user_ids:
        try:
            doc = db.collection("users").document(uid).get()
            if doc.exists:
                user_cache[uid] = doc.to_dict()
        except Exception:
            pass

    cases = []
    for obj in raw_cases:
        user = user_cache.get(obj.get("user_id"), {})
        obj["patient_email"] = user.get("email", obj.get("user_id", "Unknown"))

        if "probabilities" in obj and isinstance(obj["probabilities"], dict):
            max_class = max(obj["probabilities"], key=obj["probabilities"].get)
            confidence = round(obj["probabilities"][max_class] * 100, 2)
            obj["result_display"] = f"{obj['result']} ({confidence}%)"
        else:
            obj["result_display"] = obj.get("result", "N/A")

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

    # Collect all prediction ids first
    raw_preds = []
    pred_ids = []
    for d in preds_snap:
        obj = d.to_dict()
        obj["id"] = d.id
        raw_preds.append(obj)
        pred_ids.append(d.id)

    # Batch fetch all reviews in one query
    reviews_by_pred = {pid: [] for pid in pred_ids}
    doctor_ids = set()
    if pred_ids:
        try:
            reviews_snap = db.collection("reviews") \
                .where("prediction_id", "in", pred_ids[:30]).get()
            for r in reviews_snap:
                rdict = r.to_dict()
                pid = rdict.get("prediction_id")
                if pid in reviews_by_pred:
                    reviews_by_pred[pid].append(rdict)
                if rdict.get("doctor_id"):
                    doctor_ids.add(rdict["doctor_id"])
        except Exception as e:
            print(f"[WARN] Reviews batch fetch failed: {e}")

    # Batch fetch all doctor emails
    doctor_cache = {}
    for did in doctor_ids:
        try:
            doc = db.collection("users").document(did).get()
            if doc.exists:
                doctor_cache[did] = doc.to_dict()
        except Exception:
            pass

    # Assemble predictions with reviews
    predictions = []
    for obj in raw_preds:
        reviews = reviews_by_pred.get(obj["id"], [])
        for r in reviews:
            doctor = doctor_cache.get(r.get("doctor_id"), {})
            r["doctor_email"] = doctor.get("email", r.get("doctor_id", "Unknown"))
        obj["reviews"] = reviews
        predictions.append(obj)

    return render_template('dashboard_patient.html', predictions=predictions)


from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT


@bp.route('/dashboard/patient/export_pdf')
@role_required('patient')
def export_pdf():
    # Fetch predictions
    try:
        preds_snap = db.collection("predictions") \
            .where("user_id", "==", current_user.id) \
            .order_by("created_at", direction=firestore.Query.DESCENDING) \
            .limit(200).get()
    except Exception:
        preds_snap = db.collection("predictions") \
            .where("user_id", "==", current_user.id) \
            .limit(200).get()

    # Batch fetch reviews
    predictions = []
    pred_ids = []
    raw_preds = []
    for d in preds_snap:
        obj = d.to_dict()
        obj["id"] = d.id
        raw_preds.append(obj)
        pred_ids.append(d.id)

    reviews_by_pred = {pid: [] for pid in pred_ids}
    doctor_ids = set()
    if pred_ids:
        try:
            reviews_snap = db.collection("reviews") \
                .where("prediction_id", "in", pred_ids[:30]).get()
            for r in reviews_snap:
                rdict = r.to_dict()
                pid = rdict.get("prediction_id")
                if pid in reviews_by_pred:
                    reviews_by_pred[pid].append(rdict)
                if rdict.get("doctor_id"):
                    doctor_ids.add(rdict["doctor_id"])
        except Exception:
            pass

    doctor_cache = {}
    for did in doctor_ids:
        try:
            doc = db.collection("users").document(did).get()
            if doc.exists:
                doctor_cache[did] = doc.to_dict()
        except Exception:
            pass

    for obj in raw_preds:
        reviews = reviews_by_pred.get(obj["id"], [])
        for r in reviews:
            doctor = doctor_cache.get(r.get("doctor_id"), {})
            r["doctor_email"] = doctor.get("email", r.get("doctor_id", "Unknown"))
        obj["reviews"] = reviews
        predictions.append(obj)

    # ── Build PDF ──
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()
    PRIMARY = colors.HexColor('#1558a8')
    DANGER  = colors.HexColor('#b91c1c')
    SUCCESS = colors.HexColor('#0f7a3e')
    MUTED   = colors.HexColor('#6b7280')
    LIGHT   = colors.HexColor('#f0f4ff')

    title_style = ParagraphStyle('Title',
        fontSize=20, textColor=PRIMARY, fontName='Helvetica-Bold',
        spaceAfter=4, alignment=TA_LEFT)
    sub_style = ParagraphStyle('Sub',
        fontSize=10, textColor=MUTED, fontName='Helvetica',
        spaceAfter=2)
    section_style = ParagraphStyle('Section',
        fontSize=12, textColor=PRIMARY, fontName='Helvetica-Bold',
        spaceBefore=12, spaceAfter=6)
    body_style = ParagraphStyle('Body',
        fontSize=9, textColor=colors.HexColor('#374151'),
        fontName='Helvetica', spaceAfter=3)
    note_style = ParagraphStyle('Note',
        fontSize=8, textColor=MUTED, fontName='Helvetica-Oblique',
        spaceAfter=2)

    elements = []

    # Header
    elements.append(Paragraph('NeuroScan', title_style))
    elements.append(Paragraph('AI-Powered Ischemic Stroke Detection System', sub_style))
    elements.append(Paragraph('Prediction History Report', sub_style))
    elements.append(HRFlowable(width='100%', thickness=1.5, color=PRIMARY, spaceAfter=10))

    # Patient info
    from datetime import datetime
    now = datetime.now().strftime("%d %b %Y, %H:%M")
    info_data = [
        ['Patient', current_user.name or current_user.email],
        ['Email', current_user.email],
        ['Generated', now],
        ['Total Scans', str(len(predictions))],
    ]
    info_table = Table(info_data, colWidths=[3.5*cm, 12*cm])
    info_table.setStyle(TableStyle([
        ('FONTNAME',    (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME',    (1,0), (1,-1), 'Helvetica'),
        ('FONTSIZE',    (0,0), (-1,-1), 9),
        ('TEXTCOLOR',   (0,0), (0,-1), MUTED),
        ('TEXTCOLOR',   (1,0), (1,-1), colors.HexColor('#111827')),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.white, LIGHT]),
        ('TOPPADDING',  (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 0.4*cm))

    # Disclaimer
    elements.append(Paragraph(
        '⚠ Disclaimer: This report is generated by an AI research prototype and is NOT a substitute '
        'for professional medical diagnosis. All results should be reviewed by a qualified radiologist.',
        note_style))
    elements.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#e5e7eb'), spaceAfter=8))

    if not predictions:
        elements.append(Paragraph('No predictions found.', body_style))
    else:
        elements.append(Paragraph(f'Prediction Records ({len(predictions)} total)', section_style))

        # Table header
        table_data = [['#', 'Date', 'Model', 'AI Result', 'Confidence', 'Status', 'Reviews']]

        for i, p in enumerate(predictions, 1):
            date = p['created_at'].strftime("%d %b %Y") if p.get('created_at') else 'N/A'
            model = (p.get('model_used') or 'N/A').upper()
            result = (p.get('result') or 'N/A').capitalize()

            # Confidence
            conf = 'N/A'
            if p.get('probabilities') and isinstance(p['probabilities'], dict):
                max_prob = max(p['probabilities'].values())
                conf = f"{round(max_prob * 100, 1)}%"

            status = 'Reviewed' if p.get('status') == 'reviewed' else 'Pending'
            review_count = len(p.get('reviews', []))
            reviews_str = f"{review_count} review(s)" if review_count else "None"

            table_data.append([str(i), date, model, result, conf, status, reviews_str])

        col_widths = [0.8*cm, 2.8*cm, 3*cm, 2.2*cm, 2.2*cm, 2.2*cm, 2.8*cm]
        pred_table = Table(table_data, colWidths=col_widths, repeatRows=1)
        pred_table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND',   (0,0), (-1,0), PRIMARY),
            ('TEXTCOLOR',    (0,0), (-1,0), colors.white),
            ('FONTNAME',     (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',     (0,0), (-1,0), 8),
            ('ALIGN',        (0,0), (-1,0), 'CENTER'),
            # Body rows
            ('FONTNAME',     (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE',     (0,1), (-1,-1), 8),
            ('ALIGN',        (0,1), (-1,-1), 'CENTER'),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT]),
            ('TOPPADDING',   (0,0), (-1,-1), 5),
            ('BOTTOMPADDING',(0,0), (-1,-1), 5),
            ('GRID',         (0,0), (-1,-1), 0.3, colors.HexColor('#e5e7eb')),
        ]))
        elements.append(pred_table)

        # Colour result cells
        for i, p in enumerate(predictions, 1):
            result = p.get('result', '')
            if result == 'clot':
                pred_table.setStyle(TableStyle([
                    ('TEXTCOLOR', (3, i), (3, i), DANGER),
                    ('FONTNAME',  (3, i), (3, i), 'Helvetica-Bold'),
                ]))
            else:
                pred_table.setStyle(TableStyle([
                    ('TEXTCOLOR', (3, i), (3, i), SUCCESS),
                ]))

        # Doctor reviews detail
        has_reviews = any(p.get('reviews') for p in predictions)
        if has_reviews:
            elements.append(Spacer(1, 0.5*cm))
            elements.append(Paragraph('Doctor Reviews Detail', section_style))

            for p in predictions:
                if not p.get('reviews'):
                    continue
                date = p['created_at'].strftime("%d %b %Y") if p.get('created_at') else 'N/A'
                result = (p.get('result') or 'N/A').capitalize()
                elements.append(Paragraph(
                    f"<b>{date}</b> · {(p.get('model_used') or 'N/A').upper()} · Result: {result}",
                    body_style))
                for r in p['reviews']:
                    decision = '✔ Agrees' if r.get('decision') == 'agree' else '✖ Disagrees'
                    doc_email = r.get('doctor_email', 'Unknown')
                    notes = r.get('notes', '')
                    rev_date = r['created_at'].strftime("%d %b %Y, %H:%M") if r.get('created_at') else ''
                    line = f"&nbsp;&nbsp;&nbsp;Dr. {doc_email} — {decision}"
                    if notes:
                        line += f': "{notes}"'
                    if rev_date:
                        line += f" <font color='#9ca3af'>({rev_date})</font>"
                    elements.append(Paragraph(line, note_style))
                elements.append(Spacer(1, 0.2*cm))

    # Footer note
    elements.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#e5e7eb'), spaceBefore=12))
    elements.append(Paragraph(
        'NeuroScan FYP · Universiti Tunku Abdul Rahman · For research purposes only.',
        note_style))

    doc.build(elements)
    buffer.seek(0)

    from flask import send_file
    return send_file(
        buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'neuroscan_report_{current_user.id[:8]}.pdf'
    )

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

@bp.route('/admin/delete_user/<user_id>', methods=['POST'])
@role_required("admin")
def delete_user(user_id):
    # Prevent admin from deleting themselves
    if user_id == current_user.id:
        return jsonify({'error': 'You cannot delete your own account.'}), 400
    try:
        # Delete from Firebase Auth
        try:
            auth.delete_user(user_id)
        except Exception as e:
            print(f"[WARN] Firebase Auth delete failed (may not exist): {e}")
 
        # Delete from Firestore users collection
        db.collection("users").document(user_id).delete()
 
        # Delete all their predictions and associated reviews
        preds = db.collection("predictions").where("user_id", "==", user_id).get()
        for pred in preds:
            # Delete reviews for this prediction
            reviews = db.collection("reviews").where("prediction_id", "==", pred.id).get()
            for review in reviews:
                review.reference.delete()
            pred.reference.delete()
 
        log_event(current_user.id, current_user.role, "delete_user", {"deleted_user_id": user_id})
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
 
 
@bp.route('/admin/delete_prediction/<prediction_id>', methods=['POST'])
@role_required("admin")
def delete_prediction(prediction_id):
    try:
        # Delete all reviews linked to this prediction first
        reviews = db.collection("reviews").where("prediction_id", "==", prediction_id).get()
        for review in reviews:
            review.reference.delete()
 
        # Delete the prediction itself
        db.collection("predictions").document(prediction_id).delete()
 
        log_event(current_user.id, current_user.role, "delete_prediction", {"prediction_id": prediction_id})
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
 
    # Save review
    db.collection("reviews").add({
        "prediction_id": prediction_id,
        "doctor_id": current_user.id,
        "decision": decision,
        "notes": notes,
        "created_at": firestore.SERVER_TIMESTAMP
    })
    db.collection("predictions").document(prediction_id).update({"status": "reviewed"})
    log_event(current_user.id, "doctor", "review", {
        "prediction_id": prediction_id, "decision": decision
    })
 
    # Get prediction to find patient
    try:
        pred_doc = db.collection("predictions").document(prediction_id).get()
        if pred_doc.exists:
            pred_data = pred_doc.to_dict()
            patient_id = pred_data.get("user_id")
            model_used = pred_data.get("model_used", "N/A")
 
            if patient_id:
                # Save in-app notification to Firestore
                db.collection("notifications").add({
                    "user_id":       patient_id,
                    "prediction_id": prediction_id,
                    "doctor_id":     current_user.id,
                    "decision":      decision,
                    "notes":         notes,
                    "read":          False,
                    "created_at":    firestore.SERVER_TIMESTAMP
                })
 
                # Send email notification
                patient_doc = db.collection("users").document(patient_id).get()
                if patient_doc.exists:
                    patient_data = patient_doc.to_dict()
                    patient_email = patient_data.get("email")
                    patient_name  = patient_data.get("name")
                    if patient_email:
                        send_email_notification(
                            patient_email, patient_name,
                            decision, notes, model_used
                        )
    except Exception as e:
        print(f"[WARN] Notification failed: {e}")
 
    flash("Review submitted.", "success")
    return redirect(url_for('routes.dashboard_doctor'))

@bp.route('/notifications/read/<notif_id>', methods=['POST'])
@login_required
def mark_notification_read(notif_id):
    try:
        db.collection("notifications").document(notif_id).update({"read": True})
    except Exception:
        pass
    return jsonify({'success': True}), 200
 
 
@bp.route('/notifications/read_all', methods=['POST'])
@login_required
def mark_all_notifications_read():
    try:
        notifs = db.collection("notifications") \
            .where("user_id", "==", current_user.id) \
            .where("read", "==", False).get()
        for n in notifs:
            n.reference.update({"read": True})
    except Exception:
        pass
    return jsonify({'success': True}), 200

# -----------------------------
# Error handlers
# -----------------------------
@bp.app_errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@bp.app_errorhandler(403)
def forbidden(e):
    return render_template('403.html'), 403