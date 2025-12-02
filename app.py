# app.py
import os
import datetime
from functools import wraps
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
import numpy as np

# TensorFlow imports (optional if available)
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image as keras_image
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

# Optional requests for model download
try:
    import requests
except Exception:
    requests = None

# ---------------------------
# Flask App Config
# ---------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")

# Upload folder
UPLOAD_FOLDER = os.path.join(app.root_path, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------------
# MongoDB Atlas Config
# ---------------------------
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
DB_NAME = os.environ.get("DB_NAME", "herballink")
db = client[DB_NAME]
users_collection = db.get_collection("users")
scans_collection = db.get_collection("scans")

# ---------------------------
# Models
# ---------------------------
LEAF_MODEL_PATH = os.environ.get("LEAF_MODEL_PATH", "model1.h5")
SKIN_MODEL_PATH = os.environ.get("SKIN_MODEL_PATH", "skin_disease_model.h5")
CLASSES_TXT = os.environ.get("CLASSES_TXT", "classes.txt")

leaf_model = None
skin_model = None
disease_classes = []

leaf_class_names = [
    'Aloevera','Amruthaballi','Arali','Bhrami','Curry leaves','Doddpathre','Hibiscus',
    'Mint','Neem','Tulsi','Turmeric','Unknown'
]

if TF_AVAILABLE:
    try:
        if os.path.exists(LEAF_MODEL_PATH):
            leaf_model = load_model(LEAF_MODEL_PATH)
            app.logger.info("Leaf model loaded")
    except Exception as e:
        app.logger.warning("Leaf model load failed: %s", e)

    try:
        if os.path.exists(SKIN_MODEL_PATH):
            skin_model = load_model(SKIN_MODEL_PATH)
            app.logger.info("Skin model loaded")
    except Exception as e:
        app.logger.warning("Skin model load failed: %s", e)

    try:
        if os.path.exists(CLASSES_TXT):
            with open(CLASSES_TXT, "r") as f:
                disease_classes = [line.strip() for line in f if line.strip()]
            app.logger.info("Loaded %d disease classes", len(disease_classes))
    except Exception as e:
        app.logger.warning("Failed to load classes.txt: %s", e)

# ---------------------------
# Leaf & Skin Info
# ---------------------------
leaf_info = {
    "Aloevera": {"uses": "Soothes burns, reduces scars, hydrates skin, and promotes wound healing.", "diseases": ["Eczema", "Psoriasis", "Acne", "Skin ulcers", "Sunburn", "Dry skin"]},
    "Amruthaballi": {"uses": "Detoxifies blood and reduces skin allergies.", "diseases": ["Skin rashes", "Skin allergies"]},
    "Tulsi": {"uses": "Has antibacterial properties for skin health.", "diseases": ["Acne", "Skin infections"]},
    "Neem": {"uses": "Powerful antimicrobial, treats many skin conditions.", "diseases": ["Eczema", "Acne", "Psoriasis", "Ringworm", "Scabies", "Fungal infections"]},
    "Mint": {"uses": "Cools and refreshes skin.", "diseases": ["Acne", "Skin irritation"]},
    "Turmeric": {"uses": "Natural anti-inflammatory for skin issues.", "diseases": ["Eczema", "Psoriasis", "Skin infections", "Wounds", "Acne scars"]},
    "Ginger": {"uses": "Improves skin elasticity and reduces inflammation.", "diseases": ["Inflammatory skin conditions", "Skin aging"]},
    "Lemon": {"uses": "Lightens pigmentation and reduces acne scars.", "diseases": ["Hyperpigmentation", "Acne scars", "Oily skin"]},
    "Guava": {"uses": "Improves skin texture and prevents premature aging.", "diseases": ["Skin aging", "Wrinkles"]},
    "Henna": {"uses": "Cools skin and treats fungal issues.", "diseases": ["Fungal skin infections", "Skin rashes", "Burns"]},
    "Hibiscus": {"uses": "Rejuvenates skin and reduces aging signs.", "diseases": ["Wrinkles", "Skin aging"]},
    "Rose": {"uses": "Improves skin glow and soothes irritation.", "diseases": ["Acne", "Dry skin", "Skin irritation"]},
    "Ashoka": {"uses": "Has properties for skin rejuvenation.", "diseases": ["Skin pigmentation", "Premature aging"]},
    "Curry leaves": {"uses": "Nourishes skin and prevents dryness.", "diseases": ["Dry skin"]},
    "Arali": {"uses":"Reduces inflammation, soothes rashes and promotes wound healing.", "diseases": ["Eczema","Ringworm","Minor Wounds"]},
    "Doddpathre": {"uses":"Soothes the skin and reduces itching, redness and microbial infections.", "diseases": ["Skin rashes","Eczema","Ringworm"]},
}

recommendations = {
    "acne": "Neem leaves act as a natural antibacterial agent...",
    "eczema": "Aloe vera and neem leaves help calm itching...",
    "psoriasis": "Neem and aloe vera leaves reduce scaling...",
    "ringworm": "Basil (Tulsi) and neem leaves possess antifungal properties...",
    "unknown": "No recommendation."
}

# ---------------------------
# Helpers
# ---------------------------
def allowed_file(filename):
    return filename and "." in filename and filename.rsplit(".", 1)[1].lower() in {"png", "jpg", "jpeg"}

def predict_leaf_image(filepath):
    if leaf_model is None:
        return "Unknown", 0.0
    img = keras_image.load_img(filepath, target_size=(128,128))
    arr = keras_image.img_to_array(img)/255.0
    arr = np.expand_dims(arr, axis=0)
    preds = leaf_model.predict(arr)[0]
    idx = int(np.argmax(preds))
    conf = float(np.max(preds))*100
    name = leaf_class_names[idx] if idx < len(leaf_class_names) else "Unknown"
    if conf < 10.0:
        return "Not a Leaf", conf
    return name, conf

def predict_skin_image(filepath):
    if skin_model is None or not disease_classes:
        return "unknown", 0.0
    img = keras_image.load_img(filepath, target_size=(128,128))
    arr = keras_image.img_to_array(img)/255.0
    arr = np.expand_dims(arr, axis=0)
    preds = skin_model.predict(arr)[0]
    idx = int(np.argmax(preds))
    conf = float(np.max(preds))
    pred = disease_classes[idx] if idx < len(disease_classes) else "unknown"
    if conf < 0.05:
        return "unknown", 0.0
    return pred, conf

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            flash("Please login first", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

# ---------------------------
# Routes
# ---------------------------

@app.route("/")
def home():
    return render_template("main_page.html")

@app.route("/explore")
def explore():
    return redirect(url_for("login"))

@app.route("/scan_home")
@login_required
def scan_home():
    return render_template("scanning.html")

@app.route("/scan_leaf")
@login_required
def scan_leaf_page():
    return render_template("scan_leaf.html")

@app.route("/scan_skin")
@login_required
def scan_skin_page():
    return render_template("scan_skin.html")

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method=="POST":
        fullname = request.form.get("fullname")
        email = request.form.get("email")
        password = request.form.get("password")
        confirm = request.form.get("confirmPassword")
        if not fullname or not email or not password:
            flash("All fields required", "danger")
            return redirect(url_for("register"))
        if password != confirm:
            flash("Passwords do not match", "danger")
            return redirect(url_for("register"))
        if users_collection.find_one({"email":email}):
            flash("Email already registered", "warning")
            return redirect(url_for("register"))
        users_collection.insert_one({
            "fullname": fullname,
            "email": email,
            "password": generate_password_hash(password),
            "registered_at": datetime.datetime.utcnow()
        })
        flash("Registration successful! Login now.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method=="POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = users_collection.find_one({"email": email})
        if user and check_password_hash(user["password"], password):
            session["user"] = {"email": email, "fullname": user.get("fullname")}
            flash(f"Welcome {user.get('fullname') or email}", "success")
            return redirect(url_for("scan_home"))
        flash("Invalid credentials", "danger")
        return redirect(url_for("login"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out", "info")
    return redirect(url_for("home"))

# ---------------------------
# Prediction Endpoints
# ---------------------------
@app.route("/predict-leaf", methods=["POST"])
@login_required
def predict_leaf():
    if "image" not in request.files:
        return jsonify({"error":"no file"}),400
    file = request.files["image"]
    if file.filename=="" or not allowed_file(file.filename):
        return jsonify({"error":"invalid file"}),400
    filename = file.filename
    safe_name = f"{int(datetime.datetime.utcnow().timestamp()*1000)}_{filename}"
    save_path = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(save_path)
    leaf_name, conf = predict_leaf_image(save_path)
    info = leaf_info.get(leaf_name, {"uses":"No info","diseases":[]})
    scans_collection.insert_one({
        "type":"leaf","leaf_name":leaf_name,"uses":info["uses"],
        "diseases":info["diseases"],"confidence":conf,
        "filename":filename,"saved_name":safe_name,
        "timestamp":datetime.datetime.utcnow()
    })
    return jsonify({"leaf_name":leaf_name,"uses":info["uses"],"diseases":info["diseases"],"confidence":conf})

@app.route("/predict", methods=["POST"])
@login_required
def predict_skin():
    if "image" not in request.files:
        return jsonify({"predicted_class":"No image","confidence":0.0,"recommendation":"N/A"})
    file = request.files["image"]
    filename = file.filename
    safe_name = f"{int(datetime.datetime.utcnow().timestamp()*1000)}_{filename}"
    save_path = os.path.join(UPLOAD_FOLDER, safe_name)
    file.save(save_path)
    pred_class, conf = predict_skin_image(save_path)
    rec = recommendations.get(pred_class, "No recommendation")
    scans_collection.insert_one({
        "type":"skin","predicted_class":pred_class,
        "confidence":conf,"filename":filename,"saved_name":safe_name,
        "timestamp":datetime.datetime.utcnow()
    })
    return jsonify({"predicted_class":pred_class,"confidence":conf,"recommendation":rec})

@app.route("/scans")
@login_required
def get_scans():
    docs = list(scans_collection.find({},{"_id":0}).sort("timestamp",-1).limit(200))
    return jsonify(docs)

# ---------------------------
# Run App
# ---------------------------
if __name__=="__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0", port=port, debug=True)
