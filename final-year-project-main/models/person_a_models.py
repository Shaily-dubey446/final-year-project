"""
Bird Collision Risk Prediction - Person A's Models
------------------------------------------------------------------
Models: Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN)

Dataset: data/processed/bird_collision_model_dataset_FINAL.csv
Target : risk_target (0 = Low, 1 = Medium, 2 = High)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# STEP 1: Load the dataset
# ============================================================
df = pd.read_csv("data/processed/bird_collision_model_dataset_FINAL.csv")
print("Dataset shape:", df.shape)
print(df.head())

# ============================================================
# STEP 2: Prepare features (X) and target (y)
# ============================================================
# Drop columns that are:
#  - the target itself in text form (risk_level == risk_target, just text)
#  - identifiers / high-cardinality text fields not useful as direct features
#  - raw date (we already have flight_year / flight_month extracted from it)
drop_cols = [
    "risk_level",          # this is the same as risk_target, just as text
    "flight_date",         # raw date -> already have flight_year/flight_month
    "airport_name_clean",  # too many unique values (80+) to use directly
    "Station_ID",          # duplicate info of state_abbr
    "wildlife_species",    # very high cardinality (many species) - skip for now
]
drop_cols = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=drop_cols + ["risk_target"])
y = df["risk_target"]

print("\nFeature columns used:", X.columns.tolist())

# ============================================================
# STEP 3: Encode categorical (text) columns to numbers
# ============================================================
# Models can't understand text directly, so we convert categories
# like "Small"/"Medium"/"Large" or True/False into numbers.
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
# 80% data for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTrain size:", X_train.shape, "| Test size:", X_test.shape)

# ============================================================
# STEP 5: Scale features
# ============================================================
# Logistic Regression and KNN are distance/gradient based, so they
# work much better when all features are on a similar scale.
# (Decision Tree doesn't need this, but scaling doesn't hurt it either.)
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

    # Confusion matrix plot
    cm = confusion_matrix(y_te, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low","Medium","High"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(f"models/confusion_matrix_{name.replace(' ','_')}.png")
    plt.close()

    results[name] = acc
    return model

# ---- Model 1: Logistic Regression ----
log_reg = LogisticRegression(max_iter=1000, random_state=42)
evaluate_model("Logistic Regression", log_reg, X_train_scaled, X_test_scaled, y_train, y_test)

# ---- Model 2: Decision Tree ----
dec_tree = DecisionTreeClassifier(max_depth=8, random_state=42)
evaluate_model("Decision Tree", dec_tree, X_train, X_test, y_train, y_test)
# NOTE: Decision Tree doesn't need scaled data, so we use original X_train/X_test

# ---- Model 3: KNN ----
knn = KNeighborsClassifier(n_neighbors=5)
evaluate_model("KNN", knn, X_train_scaled, X_test_scaled, y_train, y_test)

# ============================================================
# STEP 7: Compare all 3 models
# ============================================================
print("\n" + "="*50)
print("FINAL COMPARISON (Person A's models)")
print("="*50)
for name, acc in results.items():
    print(f"{name}: {round(acc,4)}")

# Bar chart comparison
plt.figure(figsize=(6,4))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.ylabel("Accuracy")
plt.title("Model Comparison - Person A")
plt.ylim(0,1)
plt.savefig("models/person_a_comparison.png")
plt.show()

print("\nDone! Confusion matrix images and comparison chart saved in 'models/' folder.")