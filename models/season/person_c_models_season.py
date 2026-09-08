import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from xgboost import XGBClassifier


# ==============================
# CREATE SEASON MODEL FOLDER
# ==============================
os.makedirs("models/season", exist_ok=True)


# ==============================
# LOAD SEASON DATASET
# ==============================
DATA_PATH = "data/season/bird_collision_model_dataset_v2_with_season.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Could not find dataset: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(f"Dataset loaded from {DATA_PATH}")
print(f"Dataset Shape: {df.shape}")


# ==============================
# TARGET COLUMN
# ==============================
if "risk_target" not in df.columns:
    raise ValueError("risk_target column not found in dataset.")

target_col = "risk_target"


# ==============================
# HANDLE MISSING VALUES
# ==============================
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(exclude=[np.number]).columns

for col in num_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])


# ==============================
# DROP COLUMNS
# Same as Person C old script
# ==============================
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


# ==============================
# FEATURES AND TARGET
# ==============================
y = df[target_col]

X = df.drop(columns=drop_cols + [target_col])

print("\nFeatures used:")
print(X.columns.tolist())


# ==============================
# LABEL ENCODING
# ==============================
categorical_cols = X.select_dtypes(
    exclude=[np.number]
).columns.tolist()

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le


# ==============================
# TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(
    f"\nTrain size: {X_train.shape} | "
    f"Test size: {X_test.shape}"
)


# ==============================
# STANDARD SCALER
# Required for MLP
# ==============================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==============================
# SAVE PREPROCESSING FILES
# ==============================
joblib.dump(
    scaler,
    "models/season/person_c_scaler_season.pkl"
)

joblib.dump(
    X.columns.tolist(),
    "models/season/person_c_feature_columns_season.pkl"
)

joblib.dump(
    label_encoders,
    "models/season/person_c_label_encoders_season.pkl"
)


# ==============================
# PERSON C MODELS
# ==============================
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


# ==============================
# EVALUATION
# ==============================
evaluation_metrics = {}


def evaluate_and_save(
    name,
    model,
    X_tr,
    X_te,
    y_tr,
    y_te
):

    print(f"\nTraining & Evaluating: {name}")

    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)

    acc = accuracy_score(y_te, y_pred)

    prec = precision_score(
        y_te,
        y_pred,
        average="weighted",
        zero_division=0
    )

    rec = recall_score(
        y_te,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_te,
        y_pred,
        average="weighted",
        zero_division=0
    )

    evaluation_metrics[name] = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    }

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_te,
            y_pred,
            target_names=["Low", "Medium", "High"],
            zero_division=0
        )
    )

    # ==============================
    # CONFUSION MATRIX
    # ==============================
    cm = confusion_matrix(y_te, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Low", "Medium", "High"]
    )

    disp.plot(cmap="Purples")

    plt.title(
        f"Confusion Matrix - {name} (Person C Season)"
    )

    plt.tight_layout()

    file_safe_name = name.replace(" ", "_").lower()

    confusion_path = (
        f"models/season/"
        f"person_c_confusion_matrix_"
        f"{file_safe_name}_season.png"
    )

    plt.savefig(confusion_path)
    plt.close()

    print(
        f"Saved confusion matrix to: "
        f"{confusion_path}"
    )

    # ==============================
    # SAVE MODEL
    # ==============================
    model_filename = (
        f"models/season/"
        f"person_c_{file_safe_name}_"
        f"model_season.pkl"
    )

    joblib.dump(model, model_filename)

    print(
        f"Saved model to: {model_filename}"
    )


# ==============================
# TRAIN MODELS
# ==============================

# XGBoost
evaluate_and_save(
    "XGBoost",
    models["XGBoost"],
    X_train,
    X_test,
    y_train,
    y_test
)


# Gradient Boosting
evaluate_and_save(
    "Gradient Boosting",
    models["Gradient Boosting"],
    X_train,
    X_test,
    y_train,
    y_test
)


# MLP
evaluate_and_save(
    "MLP Classifier",
    models["MLP Classifier"],
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
)


# ==============================
# FINAL COMPARISON
# ==============================
print(
    "\nFINAL MODEL PERFORMANCE "
    "COMPARISON (Person C Season)"
)

metrics_df = pd.DataFrame(
    evaluation_metrics
).T

print(metrics_df.round(4))


# ==============================
# COMPARISON GRAPH
# ==============================
ax = metrics_df.plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title(
    "Model Performance Comparison - "
    "Person C Season",
    fontsize=14,
    fontweight="bold"
)

plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.7
)

plt.legend(loc="lower right")

plt.tight_layout()

comparison_path = (
    "models/season/"
    "person_c_comparison_season.png"
)

plt.savefig(comparison_path)
plt.close()


# ==============================
# FINAL MESSAGE
# ==============================
print(
    "\nAll Person C Season Models "
    "and Artifacts Saved Successfully!"
)

print("\nSaved files inside models/season/:")
print(" - person_c_xgboost_model_season.pkl")
print(" - person_c_gradient_boosting_model_season.pkl")
print(" - person_c_mlp_classifier_model_season.pkl")
print(" - person_c_scaler_season.pkl")
print(" - person_c_label_encoders_season.pkl")
print(" - person_c_feature_columns_season.pkl")
print(" - Person C confusion matrices")
print(" - person_c_comparison_season.png")