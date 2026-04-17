from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import joblib
import pandas as pd
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "your_secret_key_here"

MODEL_PATH = "saved_models/cnn1d_model.h5"
FEATURE_PATH = "saved_models/cnn1d_feature_columns.pkl"

model = load_model(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)


# -----------------------
# Helper Function
# -----------------------
def get_files(folder):
    files = []
    path = os.path.join("static", folder)
    if os.path.exists(path):
        files = os.listdir(path)
    return files

# -----------------------
# Routes
# -----------------------

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":

        file = request.files["dataset"]

        if file and file.filename != "":
            filename = file.filename
            filepath = os.path.join("uploads", filename)
            file.save(filepath)

            session["dataset_path"] = filepath   # 🔥 Store in session

            flash("Dataset uploaded successfully!", "success")
            return redirect(url_for("upload"))

    return render_template("upload.html")


@app.route("/eda")
def eda():
    images = get_files("vis")
    return render_template("eda.html", images=images)

@app.route("/ml_performance")
def ml_performance():
    files = get_files("performance")

    images = [f for f in files if f.endswith(".png")]
    reports = [f for f in files if f.endswith(".txt") or f.endswith(".csv")]

    return render_template("ml_performance.html",
                           images=images,
                           reports=reports)


@app.route("/dl_performance")
def dl_performance():
    files = get_files("dl_performance")

    images = [f for f in files if f.endswith(".png")]
    reports = [f for f in files if f.endswith(".txt") or f.endswith(".csv")]

    return render_template("dl_performance.html",
                           images=images,
                           reports=reports)


@app.route("/best_model")
def best_model():
    files = get_files("dl_performance")

    cnn1d_files = [f for f in files if "CNN1D" in f]
    images = [f for f in cnn1d_files if f.endswith(".png")]
    reports = [f for f in cnn1d_files if f.endswith(".txt") or f.endswith(".csv")]

    return render_template("best_model.html",
                           images=images,
                           reports=reports)

@app.route("/predict_customer", methods=["GET", "POST"])
def predict_customer():

    if "dataset_path" not in session:
        flash("Please upload dataset first!", "warning")
        return redirect(url_for("upload"))

    # =============================
    # LOAD DATASET
    # =============================
    df = pd.read_csv(session["dataset_path"])

    if "CONS_NO" not in df.columns:
        flash("CONS_NO column not found in dataset!", "danger")
        return redirect(url_for("upload"))

    prediction_result = None
    probability = None
    risk_level = None

    if request.method == "POST":

        customer_id = request.form.get("customer_id")

        if not customer_id:
            flash("Please enter Customer ID", "warning")
            return redirect(request.url)

        customer_id = customer_id.strip()

        # Ensure CONS_NO handled as string safely
        df["CONS_NO"] = df["CONS_NO"].astype(str).str.strip()

        # =============================
        # MATCH CUSTOMER
        # =============================
        customer_row = df[df["CONS_NO"].str.upper() == customer_id.upper()]

        if customer_row.empty:
            flash("Customer not found!", "danger")
            return redirect(request.url)

        # =============================
        # PREPROCESS SAME AS TRAINING
        # =============================

        # Separate ID
        cons_no_series = customer_row["CONS_NO"]

        # Drop ID + target
        X = customer_row.drop(columns=["FLAG", "CONS_NO"], errors="ignore")

        # Convert columns to datetime (same as training)
        X.columns = pd.to_datetime(X.columns)

        # Sort columns (same as training)
        X = X.reindex(sorted(X.columns), axis=1)

        # =============================
        # ALIGN WITH TRAINING FEATURES
        # =============================

        # Reindex using saved feature order
        # Missing columns will become NaN (handled below)
        X = X.reindex(columns=feature_columns)

        # Fill any missing columns with 0
        X = X.fillna(0)

        # Convert to numpy float
        X = X.to_numpy(dtype=float)

        # Reshape for CNN1D
        X = X.reshape(1, X.shape[1], 1)

        # =============================
        # PREDICTION
        # =============================
        prob = model.predict(X)[0][0]
        prediction = 1 if prob > 0.5 else 0

        prediction_result = "⚠ Theft Detected" if prediction == 1 else "Normal Usage"
        probability = round(float(prob), 4)

        # Risk Level Classification
        if prob > 0.8:
            risk_level = "High"
        elif prob > 0.5:
            risk_level = "Medium"
        else:
            risk_level = "Low"

    return render_template("predict.html",
                           prediction=prediction_result,
                           probability=probability,
                           risk_level=risk_level)





if __name__ == "__main__":
    app.run(debug=True)
