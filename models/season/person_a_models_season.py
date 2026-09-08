"""
Bird Collision Risk Prediction - Person A's Models (v2, WITH season)
------------------------------------------------------------------
Same 3 models as before (Logistic Regression, Decision Tree, KNN),
but trained on the NEW dataset that includes the 'season' feature.

This is kept completely separate from the original models/person_a_models.py
so both versions (with/without season) can be compared side by side.

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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

# Make sure the output folder exists
os.makedirs("models/season", exist_ok=True)

# ============================================================
# STEP 1: Load the dataset (NOTE: path points to data/season/)
# ============================================================
df = pd.read_csv("data/processed/season/bird_collision_model_dataset_v2_with_season.csv")
print("Dataset shape:", df.shape)
print(df.head())

# ============================================================
# STEP 2: Prepare features (X) and target (y)
# ============================================================
drop_cols = [
    "risk_level", "flight_date", "airport_name_clean",
    "Station_ID", "wildlife_species",
]
drop_cols = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=drop_cols + ["risk_target"])
y = df["risk_target"]

print("\nFeature columns used (should include 'season'):", X.columns.tolist())

# ============================================================
# STEP 3: Encode categorical (text) columns to numbers
# ============================================================
categorical_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
print("\nCategorical columns being encoded:", categorical_cols)

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# ============================================================
# STEP 4: Train-test split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTrain size:", X_train.shape, "| Test size:", X_test.shape)

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
    print(f"\n{'='*50}")
    print(f"MODEL: {name}")
    print(f"{'='*50}")
    print("Accuracy:", round(acc, 4))
    print("\nClassification Report:")
    print(classification_report(y_te, y_pred, target_names=["Low", "Medium", "High"]))

    cm = confusion_matrix(y_te, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low","Medium","High"])
    disp.plot(cmap="Purples")
    plt.title(f"Confusion Matrix - {name} (with season)")
    plt.savefig(f"models/season/confusion_matrix_{name.replace(' ','_')}.png")
    plt.close()

    results[name] = acc
    return model

log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg = evaluate_model("Logistic Regression", log_reg, X_train_scaled, X_test_scaled, y_train, y_test)

dec_tree = DecisionTreeClassifier(max_depth=8, random_state=42)
dec_tree = evaluate_model("Decision Tree", dec_tree, X_train, X_test, y_train, y_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn = evaluate_model("KNN", knn, X_train_scaled, X_test_scaled, y_train, y_test)

# ============================================================
# STEP 7: Compare all 3 models
# ============================================================
print("\n" + "="*50)
print("FINAL COMPARISON (Person A's models WITH season)")
print("="*50)
for name, acc in results.items():
    print(f"{name}: {round(acc,4)}")

plt.figure(figsize=(6,4))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.ylabel("Accuracy")
plt.title("Model Comparison - Person A (with season)")
plt.ylim(0,1)
plt.savefig("models/season/person_a_comparison_season.png")
plt.show()

# ============================================================
# STEP 8: Save the trained models + scaler + encoders
# ============================================================
with open("models/season/logistic_regression_model_season.pkl", "wb") as f:
    pickle.dump(log_reg, f)
with open("models/season/decision_tree_model_season.pkl", "wb") as f:
    pickle.dump(dec_tree, f)
with open("models/season/knn_model_season.pkl", "wb") as f:
    pickle.dump(knn, f)
with open("models/season/scaler_season.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("models/season/label_encoders_season.pkl", "wb") as f:
    pickle.dump(label_encoders, f)
with open("models/season/feature_columns_season.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("\nDone! Results saved in 'models/season/' folder.")
print("\nCOMPARE THIS to the original (without season) results:")
print("  Original - Logistic Regression: 0.7785 | Decision Tree: 0.8339 | KNN: 0.6779")