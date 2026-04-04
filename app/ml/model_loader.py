import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os
import timm
from transformers import ViTForImageClassification, AutoImageProcessor, ViTConfig
from .gradcam import GradCAM
import cv2, uuid
import numpy as np
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model_files')
CLASSES_PATH = os.path.join(MODEL_DIR, 'classes.json')

with open(CLASSES_PATH, 'r') as f:
    data = json.load(f)
classes = data['classes']  # ["clot", "no clot"]
num_classes = len(classes)

cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"

# -----------------------------
# SPEED FIX 1: Cache ViT processor at startup instead of downloading on every prediction
# Previously: AutoImageProcessor.from_pretrained() was called inside predict_stroke_risk()
# on every single ViT prediction — this caused ~300-500ms overhead each time.
# -----------------------------
print("[INFO] Loading ViT processor at startup...")
VIT_PROCESSOR = AutoImageProcessor.from_pretrained(VIT_MODEL_NAME, use_fast=True)
print("[INFO] ViT processor ready.")

# -----------------------------
# Model Loader
# -----------------------------
def load_model(model_name, checkpoint_filename):
    checkpoint_path = os.path.join(MODEL_DIR, checkpoint_filename)

    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet101":
        model = models.resnet101(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)

    elif model_name == "densenet169":
        model = models.densenet169(weights=None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)

    elif model_name == "efficientnetb3":
        model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=num_classes)

    elif model_name == "vit":
        config = ViTConfig.from_pretrained(
            VIT_MODEL_NAME,
            num_labels=num_classes,
            output_attentions=True
        )
        model = ViTForImageClassification(config)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

    else:
        raise ValueError(f"Model {model_name} not supported!")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    return model

# Load all models at startup
MODELS = {
    "resnet50":       load_model("resnet50",       "resnet50_best.pth"),
    "resnet101":      load_model("resnet101",      "resnet101_best.pth"),
    "densenet121":    load_model("densenet121",    "densenet121_best.pth"),
    "densenet169":    load_model("densenet169",    "densenet169_best.pth"),
    "efficientnetb3": load_model("efficientnetb3", "efficientnetb3_best.pth"),
    "vit":            load_model("vit",            "vit_best.pth"),
}

# -----------------------------
# SPEED FIX 2: Cache GradCAM objects per model at startup
# Previously: GradCAM(model, model_name) was instantiated inside every prediction call,
# which registers new hooks on the model every time — wasteful and slow.
# Now we create one GradCAM object per CNN model and reuse it.
# -----------------------------
CNN_MODELS = ["resnet50", "resnet101", "densenet121", "densenet169", "efficientnetb3"]
GRADCAM_CACHE = {
    name: GradCAM(MODELS[name], name) for name in CNN_MODELS
}
print("[INFO] GradCAM objects cached for all CNN models.")

# -----------------------------
# Prediction Function
# -----------------------------
def predict_stroke_risk(image_path, model_name="resnet50"):

    if model_name == "ensemble":
        return predict_ensemble(image_path)

    if model_name not in MODELS:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(MODELS.keys())}")

    model = MODELS[model_name]
    model.eval()
    image = Image.open(image_path).convert('RGB')

    gradcam_url, orig_url = None, None

    # ----------------------------
    # ViT branch (Attention Rollout)
    # ----------------------------
    if model_name == "vit":
        from .vit_rollout import VitAttentionRollout

        # SPEED FIX 1 applied: use cached VIT_PROCESSOR instead of downloading fresh
        inputs = VIT_PROCESSOR(images=image, return_tensors="pt")
        input_tensor = inputs["pixel_values"].to(device)
        rgb_img = np.array(image.resize((224, 224))).astype(np.float32) / 255.0

        with torch.no_grad():
            output = model(input_tensor)
            logits = output.logits
            probabilities = torch.nn.functional.softmax(logits[0], dim=0)
            confidence = probabilities.cpu().numpy()

        pred_idx = torch.argmax(probabilities).item()
        pred_class = classes[pred_idx]

        try:
            rollout = VitAttentionRollout(model)
            cam = rollout(input_tensor)

            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = np.uint8(0.3 * heatmap + 0.7 * (rgb_img * 255))

            cam_filename = f"vit_rollout_{uuid.uuid4().hex}.jpg"
            cam_path = os.path.join("app", "static", "uploads", cam_filename)
            os.makedirs(os.path.dirname(cam_path), exist_ok=True)
            cv2.imwrite(cam_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            gradcam_url = f"/static/uploads/{cam_filename}"

            orig_filename = f"orig_vit_{uuid.uuid4().hex}.jpg"
            orig_path = os.path.join("app", "static", "uploads", orig_filename)
            cv2.imwrite(orig_path, cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            orig_url = f"/static/uploads/{orig_filename}"

        except Exception as e:
            print(f"[WARN] ViT interpretability failed: {e}")

    # ----------------------------
    # CNN branch (Grad-CAM)
    # ----------------------------
    else:
        input_tensor = cnn_transform(image).unsqueeze(0).to(device)
        rgb_img = np.array(image.resize((224, 224))).astype(np.float32) / 255.0

        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.nn.functional.softmax(logits[0], dim=0)
            confidence = probabilities.cpu().numpy()

        pred_idx = torch.argmax(probabilities).item()
        pred_class = classes[pred_idx]

        try:
            # SPEED FIX 2 applied: reuse cached GradCAM object instead of creating new one
            gradcam = GRADCAM_CACHE[model_name]
            cam = gradcam.generate(input_tensor, target_class=pred_idx)

            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = np.uint8(0.3 * heatmap + 0.7 * (rgb_img * 255))

            cam_filename = f"gradcam_{model_name}_{uuid.uuid4().hex}.jpg"
            cam_path = os.path.join("app", "static", "uploads", cam_filename)
            cv2.imwrite(cam_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            gradcam_url = f"/static/uploads/{cam_filename}"

            orig_filename = f"orig_{model_name}_{uuid.uuid4().hex}.jpg"
            orig_path = os.path.join("app", "static", "uploads", orig_filename)
            cv2.imwrite(orig_path, cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            orig_url = f"/static/uploads/{orig_filename}"

        except Exception as e:
            print(f"[WARN] Grad-CAM failed: {e}")

    return {
        "predicted_class": pred_class,
        "probabilities": {classes[i]: float(confidence[i]) for i in range(len(classes))},
        "gradcam_path": gradcam_url,
        "original_path": orig_url
    }


# Load ensemble descriptor
ENSEMBLE_DESCRIPTOR_PATH = os.path.join(MODEL_DIR, "ensemble_best.pth")
ENSEMBLE = None
if os.path.exists(ENSEMBLE_DESCRIPTOR_PATH):
    ENSEMBLE = torch.load(ENSEMBLE_DESCRIPTOR_PATH, map_location=device, weights_only=False)


def predict_ensemble(image_path):
    """Run ensemble prediction with parallelised model inference."""
    if ENSEMBLE is None:
        raise ValueError("No ensemble_best.pth found!")

    method = ENSEMBLE["method"]
    selected_models = ENSEMBLE["selected_models"]

    image = Image.open(image_path).convert('RGB')
    rgb_img = np.array(image.resize((224, 224))).astype(np.float32) / 255.0

    # -----------------------------
    # SPEED FIX 3: Run all ensemble models in parallel using ThreadPoolExecutor
    # Previously: models ran one after another (sequential) — total time = sum of all models
    # Now: models run simultaneously — total time ≈ slowest single model
    # Note: ThreadPoolExecutor works well here because PyTorch releases the GIL during inference
    # -----------------------------
    results = {}
    with ThreadPoolExecutor(max_workers=len(selected_models)) as executor:
        future_to_name = {
            executor.submit(predict_stroke_risk, image_path, name): name
            for name in selected_models
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results[name] = future.result()
            except Exception as e:
                print(f"[WARN] Ensemble model {name} failed: {e}")
                results[name] = None

    # Collect results in original model order
    probs_list = []
    heatmaps = []
    for name in selected_models:
        res = results.get(name)
        if res is None:
            continue
        probs = res["probabilities"]
        probs_tensor = torch.tensor([probs[c] for c in classes], dtype=torch.float32)
        probs_list.append(probs_tensor)

        if res["gradcam_path"]:
            cam = cv2.imread(os.path.join("app", res["gradcam_path"].lstrip("/")), cv2.IMREAD_COLOR)
            if cam is not None:
                cam = cv2.resize(cam, (224, 224))
                heatmaps.append(cam.astype(np.float32) / 255.0)

    if not probs_list:
        raise ValueError("All ensemble models failed.")

    probs_stack = torch.stack(probs_list)

    if method == "simple_avg":
        probs_final = probs_stack.mean(dim=0)

    elif method == "weighted_avg":
        weights = torch.tensor(ENSEMBLE["weights"], dtype=torch.float32)
        probs_final = torch.matmul(weights, probs_stack)

    elif method == "stacking":
        meta = joblib.load(ENSEMBLE["meta_joblib"])
        flat = np.concatenate([p.numpy() for p in probs_list], axis=0).reshape(1, -1)
        pred_idx = meta.predict(flat)[0]
        pred_class = classes[pred_idx]
        return {
            "predicted_class": pred_class,
            "probabilities": {classes[i]: float(meta.predict_proba(flat)[0][i]) for i in range(len(classes))},
            "gradcam_path": None,
            "original_path": None
        }

    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    pred_idx = torch.argmax(probs_final).item()
    pred_class = classes[pred_idx]

    gradcam_url, orig_url = None, None
    if heatmaps:
        avg_heatmap = np.mean(heatmaps, axis=0)
        avg_heatmap = np.uint8(255 * avg_heatmap)
        overlay = np.uint8(0.3 * avg_heatmap + 0.7 * (rgb_img * 255))

        cam_filename = f"ensemble_gradcam_{uuid.uuid4().hex}.jpg"
        cam_path = os.path.join("app", "static", "uploads", cam_filename)
        os.makedirs(os.path.dirname(cam_path), exist_ok=True)
        cv2.imwrite(cam_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        gradcam_url = f"/static/uploads/{cam_filename}"

        orig_filename = f"orig_ensemble_{uuid.uuid4().hex}.jpg"
        orig_path = os.path.join("app", "static", "uploads", orig_filename)
        cv2.imwrite(orig_path, cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        orig_url = f"/static/uploads/{orig_filename}"

    return {
        "predicted_class": pred_class,
        "probabilities": {classes[i]: float(probs_final[i]) for i in range(len(classes))},
        "gradcam_path": gradcam_url,
        "original_path": orig_url
    }