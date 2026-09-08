"""
Bird Collision Risk Prediction - Person B's Models (v2, WITH season)
------------------------------------------------------------------
Same NEW dataset as Person A's season version.

Person B models:
1. Random Forest
2. Naive Bayes
3. SVM

Dataset: data/season/bird_collision_model_dataset_v2_with_season.csv
Target : risk_target (0 = Low, 1 = Medium, 2 = High)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,precision_score,recall_score, f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Make sure output folder exists
os.makedirs("models/season", exist_ok=True)

# ============================================================
# STEP 1: Load the NEW dataset
# ============================================================

df = pd.read_csv(
    "data/season/bird_collision_model_dataset_v2_with_season.csv"
)

print("Dataset shape:", df.shape)
print(df.head())

# ============================================================
# STEP 2: Prepare features (X) and target (y)
# ============================================================

drop_cols = [
    "risk_level",
    "number_struck",
    "number_struck_actual",
    "aircraft_type",
    "flight_date",
    "state_abbr",
    "Station_ID",
    "airport_name_clean",
    "altitude_bin"
]

drop_cols = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=drop_cols + ["risk_target"])
y = df["risk_target"]

print("\nFeature columns used (should include 'season'):")
print(X.columns.tolist())

# ============================================================
# STEP 3: Encode categorical columns
# ============================================================

categorical_cols = X.select_dtypes(
    include=["object", "bool"]
).columns.tolist()

print("\nCategorical columns being encoded:")
print(categorical_cols)

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# ============================================================
# STEP 4: Train-test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(
    "\nTrain size:",
    X_train.shape,
    "| Test size:",
    X_test.shape
)

# ============================================================
# STEP 5: Scale features
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# STEP 6: Train and evaluate each model
# ============================================================

results = {}

def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):

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

    print(f"\n{'='*50}")
    print(f"MODEL: {name}")
    print(f"{'='*50}")

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

    cm = confusion_matrix(y_te, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Low", "Medium", "High"]
    )

    disp.plot(cmap="Purples")

    plt.title(
        f"Confusion Matrix - Person B - {name} (with season)"
    )

    safe_name = name.replace(" ", "_")

    plt.savefig(
        f"models/season/person_b_confusion_matrix_{safe_name}_season.png"
    )

    plt.close()

    results[name] = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    }

    return model





# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------

random_forest = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

random_forest = evaluate_model(
    "Random Forest",
    random_forest,
    X_train,
    X_test,
    y_train,
    y_test
)

# ------------------------------------------------------------
# Naive Bayes
# ------------------------------------------------------------

naive_bayes = GaussianNB()

naive_bayes = evaluate_model(
    "Naive Bayes",
    naive_bayes,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
)

# ------------------------------------------------------------
# SVM
# ------------------------------------------------------------

svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42
)

svm = evaluate_model(
    "SVM",
    svm,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
)

# ============================================================
# STEP 7: Compare all 3 models
# ============================================================

print("\n" + "="*50)
print("FINAL COMPARISON (Person B's models WITH season)")
print("="*50)

metrics_df = pd.DataFrame(results).T

print(metrics_df.round(4))

plt.figure(figsize=(10, 6))

metrics_df.plot(
    kind="bar",
    figsize=(10, 5)
)

plt.ylabel("Score")
plt.title("Model Performance Comparison - Person B (with season)")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.7
)

plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    "models/season/person_b_comparison_season.png"
)

plt.close()

# ============================================================
# STEP 8: Save trained models
# ============================================================

with open(
    "models/season/person_b_random_forest_model_season.pkl",
    "wb"
) as f:
    pickle.dump(random_forest, f)

with open(
    "models/season/person_b_naive_bayes_model_season.pkl",
    "wb"
) as f:
    pickle.dump(naive_bayes, f)

with open(
    "models/season/person_b_svm_model_season.pkl",
    "wb"
) as f:
    pickle.dump(svm, f)

with open(
    "models/season/person_b_scaler_season.pkl",
    "wb"
) as f:
    pickle.dump(scaler, f)

with open(
    "models/season/person_b_label_encoders_season.pkl",
    "wb"
) as f:
    pickle.dump(label_encoders, f)

with open(
    "models/season/person_b_feature_columns_season.pkl",
    "wb"
) as f:
    pickle.dump(X.columns.tolist(), f)

print("\nDone! All Person B season results saved in:")
print("models/season/")