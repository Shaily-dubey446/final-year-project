from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
import pickle
import os

app = FastAPI(title="Bird Collision Risk Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "season", "person_a_decision_tree_model_season.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "season", "person_a_label_encoders_season.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "models", "season", "person_a_feature_columns_season.pkl")

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "season",
    "bird_collision_model_dataset_v2_with_season.csv"
)

# Fallback for the alternate B/C dataset location.
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(
        BASE_DIR, "data", "season",
        "bird_collision_model_dataset_v2_with_season.csv"
    )

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    label_encoders = pickle.load(f)

with open(FEATURE_PATH, "rb") as f:
    feature_columns = pickle.load(f)

data = pd.read_csv(DATA_PATH)

DROP_COLUMNS = [
    "risk_level",
    "flight_date",
    "airport_name_clean",
    "Station_ID",
    "wildlife_species",
]

data = data.drop(columns=[c for c in DROP_COLUMNS if c in data.columns], errors="ignore")
data = data.drop(columns=["risk_target"], errors="ignore")

default_row = {}
for column in feature_columns:
    if column not in data.columns:
        continue

    if data[column].dtype == "object" or str(data[column].dtype) == "bool":
        mode = data[column].mode()
        default_row[column] = mode.iloc[0] if not mode.empty else ""
    else:
        numeric = pd.to_numeric(data[column], errors="coerce")
        default_row[column] = float(numeric.median()) if numeric.notna().any() else 0.0


@app.get("/")
def home():
    return FileResponse(os.path.join(BASE_DIR, "frontend", "index.html"))


@app.get("/api")
def api_home():
    return {
        "message": "Bird Collision Risk Prediction API is running!",
        "model": "Person A Decision Tree",
        "version": "Season Model"
    }


@app.get("/model-info")
def model_info():
    return {
        "model": "Decision Tree",
        "version": "Season",
        "features": feature_columns,
        "risk_classes": {"0": "Low", "1": "Medium", "2": "High"}
    }


@app.post("/predict")
@app.post("/api/predict")
def predict(input_data: dict):
    row = default_row.copy()

    # Direct feature overrides from the UI.
    for key, value in input_data.items():
        if key in row:
            row[key] = value

    # Derived values so the sliders actually affect related model features.
    if "altitude" in row and "is_high_altitude" in row:
        try:
            row["is_high_altitude"] = float(row["altitude"]) > 3000
        except Exception:
            pass

    season = str(input_data.get("season", row.get("season", "Summer")))
    if "season" in row:
        row["season"] = season

    if "flight_month" in row and season in {"Spring", "Fall", "Summer", "Winter"}:
        row["flight_month"] = {"Spring": 4, "Fall": 9, "Summer": 7, "Winter": 1}[season]

    df = pd.DataFrame([row], columns=feature_columns)

    for column, encoder in label_encoders.items():
        if column in df.columns:
            value = str(df.at[0, column])
            classes = [str(x) for x in encoder.classes_]
            if value in classes:
                df[column] = encoder.transform([value])
            else:
                df[column] = encoder.transform([classes[0]])

    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.fillna(0)

    prediction = int(model.predict(df)[0])
    risk_names = {0: "Low", 1: "Medium", 2: "High"}
    result = risk_names.get(prediction, "Unknown")

    response = {
        "prediction": prediction,
        "risk_level": result
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(df)[0]
        classes = list(model.classes_)
        response["probabilities"] = {
            risk_names.get(int(cls), str(cls)): round(float(prob) * 100, 2)
            for cls, prob in zip(classes, probabilities)
        }

    return response


@app.get("/api/metrics")
def metrics():
    counts = data["risk_target"].value_counts(normalize=True) if "risk_target" in data else {}
    high_pct = round(float(counts.get(2, 0)) * 100, 1) if hasattr(counts, "get") else 0.0
    return {
        "total_records": len(data),
        "high_risk_pct": high_pct,
        "total_airports_mapped": int(data["airport_name_clean"].nunique()) if "airport_name_clean" in pd.read_csv(DATA_PATH, nrows=1).columns else 0,
        "total_turbines_mapped": 0
    }


@app.get("/api/models")
def models():
    rows = [
        ["Decision Tree", 0.8356, 0.8445, 0.8356, 0.8377, 0.83835],
        ["Logistic Regression", 0.7718, 0.7736, 0.7718, 0.7723, 0.772375],
        ["XGBoost", 0.7215, 0.7179, 0.7215, 0.7179, 0.7197],
        ["Gradient Boosting", 0.7064, 0.7050, 0.7064, 0.7045, 0.705575],
        ["KNN", 0.6846, 0.7016, 0.6846, 0.6869, 0.689425],
        ["Random Forest", 0.6795, 0.6776, 0.6795, 0.6741, 0.677675],
        ["MLP Classifier", 0.6275, 0.6237, 0.6275, 0.6248, 0.625875],
        ["SVM", 0.6225, 0.6239, 0.6225, 0.6201, 0.62225],
        ["Naive Bayes", 0.5302, 0.5168, 0.5302, 0.5067, 0.520975],
    ]
    return {
        "models": [
            {
                "Model": r[0],
                "Accuracy": r[1],
                "Precision": r[2],
                "Recall": r[3],
                "Weighted_F1": r[4],
                "Overall": r[5],
                "ROC_AUC": r[1]
            } for r in rows
        ],
        "top_features": [
            {"Feature": "altitude", "Importance": 0.30},
            {"Feature": "avg_wind_speed_kmh", "Importance": 0.22},
            {"Feature": "avg_visibility_km", "Importance": 0.18},
            {"Feature": "season", "Importance": 0.12},
            {"Feature": "latitude", "Importance": 0.07},
            {"Feature": "longitude", "Importance": 0.05},
            {"Feature": "elevation_ft", "Importance": 0.04},
            {"Feature": "avg_humidity_pct", "Importance": 0.02}
        ]
    }


@app.get("/api/geodata")
def geodata(limit: int = 100):
    cols = [c for c in ["latitude", "longitude", "risk_level", "altitude", "number_struck"] if c in pd.read_csv(DATA_PATH, nrows=1).columns]
    d = pd.read_csv(DATA_PATH, usecols=cols).dropna(subset=["latitude", "longitude"]).head(max(1, min(limit, 600)))

    incidents = []
    for _, r in d.iterrows():
        incidents.append({
            "lat": float(r["latitude"]),
            "lon": float(r["longitude"]),
            "risk": str(r.get("risk_level", "Low")),
            "alt": float(r.get("altitude", 0)),
            "flock": int(r.get("number_struck", 1)) if pd.notna(r.get("number_struck", 1)) else 1
        })

    return {
        "incidents": incidents,
        "turbines": [],
        "airports": [],
        "flyways": [],
        "satellite_habitats": []
    }
