import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

os.makedirs("models", exist_ok=True)

possible_paths = [
    "data/processed/bird_collision_model_dataset_FINAL.csv",
    "../data/processed/bird_collision_model_dataset_FINAL.csv",
    "data/processed_data.csv",
    "../data/processed_data.csv",
    os.path.join(os.path.dirname(__file__), "../data/processed/bird_collision_model_dataset_FINAL.csv"),
    os.path.join(os.path.dirname(__file__), "../../data/processed/bird_collision_model_dataset_FINAL.csv"),
]

DATA_PATH = None
for p in possible_paths:
    if os.path.exists(p):
        DATA_PATH = os.path.abspath(p)
        break

if DATA_PATH is None:
    raise FileNotFoundError(f"Could not find dataset. Checked paths: {possible_paths}")

df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded from {DATA_PATH}. Shape: {df.shape}")
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(exclude=[np.number]).columns

for col in num_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

drop_cols = [
    "risk_level",
    "number_struck",
    "number_struck_actual",
    "aircraft_type",
    "flight_date",
    "state_abbr",
    "Station_ID",
    "airport_name_clean",
    "altitude_bin",
]
drop_cols = [c for c in drop_cols if c in df.columns]

if "risk_target" in df.columns:
    target_col = "risk_target"
elif "risk_level" in df.columns:
    target_col = "risk_level"
else:
    raise ValueError("Target column (risk_target or risk_level) not found in dataset.")

y = df[target_col]
if y.dtype == object:
    risk_mapping = {"Low": 0, "Medium": 1, "High": 2}
    y = y.map(risk_mapping)

X = df.drop(columns=drop_cols + [target_col])
print("Features used:", X.columns.tolist())

categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape} | Test size: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")

models = {
    "XGBoost": XGBClassifier(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=5,
        random_state=42,
        eval_metric="mlogloss"
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=4,
        random_state=42
    ),
    "MLP Classifier": MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
        early_stopping=True
    )
}

evaluation_metrics = {}

def evaluate_and_save(name, model, X_tr, X_te, y_tr, y_te, is_scaled=False):
    print(f"\nTraining & Evaluating: {name}")
    
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_te, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_te, y_pred, average="weighted", zero_division=0)
    
    evaluation_metrics[name] = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    }
    
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f} (Weighted)")
    print(f"Recall   : {rec:.4f} (Weighted)")
    print(f"F1-Score : {f1:.4f} (Weighted)")
    print("Classification Report:")
    print(classification_report(y_te, y_pred, target_names=["Low", "Medium", "High"], zero_division=0))
    
    cm = confusion_matrix(y_te, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low", "Medium", "High"])
    disp.plot(cmap="Purples")
    plt.title(f"Confusion Matrix - {name} (Person C)")
    file_safe_name = name.replace(" ", "_").lower()
    plt.tight_layout()
    plt.savefig(f"models/confusion_matrix_{file_safe_name}.png")
    plt.close()
    
    model_filename = f"models/{file_safe_name}_model.pkl"
    joblib.dump(model, model_filename)
    print(f"Saved model to: {model_filename}")
    
    return model

evaluate_and_save("XGBoost", models["XGBoost"], X_train, X_test, y_train, y_test)
evaluate_and_save("Gradient Boosting", models["Gradient Boosting"], X_train, X_test, y_train, y_test)
evaluate_and_save("MLP Classifier", models["MLP Classifier"], X_train_scaled, X_test_scaled, y_train, y_test, is_scaled=True)

print("\nFINAL MODEL PERFORMANCE COMPARISON (Person C)")
metrics_df = pd.DataFrame(evaluation_metrics).T
print(metrics_df.round(4))

plt.figure(figsize=(10, 6))
metrics_df.plot(kind="bar", figsize=(10, 5), colormap="viridis")
plt.title("Model Performance Comparison - Person C", fontsize=14, fontweight="bold")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("models/person_c_comparison.png")
plt.close()

print("\nArtifacts Saved Successfully in `models/`:")
print(" - `scaler.pkl`")
print(" - `feature_columns.pkl`")
print(" - `label_encoders.pkl`")
print(" - `xgboost_model.pkl`")
print(" - `gradient_boosting_model.pkl`")
print(" - `mlp_classifier_model.pkl`")
print(" - Confusion matrix & comparison charts (.png)")
