"""
Bird Strike Risk Prediction
===========================
Compares three classifiers - Random Forest, Naive Bayes, and SVM - on the
task of predicting `risk_target` (0 = Low, 1 = Medium, 2 = High) for a
bird-aircraft collision, using conditions known at/around the time of the
flight (weather, altitude, aircraft, location, wildlife type, etc).

Run:
    python bird_collision_models.py --data bird_collision_model_dataset_FINAL.csv
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, f1_score
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# 2. Clean / select features
# ---------------------------------------------------------------------------
def prepare_features(df: pd.DataFrame):
    """
    Drops columns that are either:
      - the target itself in disguise (risk_level == risk_target as text)
      - direct leakage (number_struck / number_struck_actual essentially
        define the risk level, so a model trained on them would be cheating)
      - constant / redundant with other columns already kept
        (aircraft_type is always "Airplane"; state_abbr, Station_ID,
        airport_name_clean duplicate origin_state / lat-long;
        altitude_bin is a binned copy of altitude; flight_date is already
        broken out into flight_year / flight_month)
    """
    target_col = "risk_target"

    drop_cols = [
        "risk_level",              # duplicate of the target (text label)
        "number_struck",           # leakage: defines the target
        "number_struck_actual",    # leakage: defines the target
        "aircraft_type",           # constant, single value
        "flight_date",             # redundant with flight_year/flight_month
        "state_abbr",              # redundant with origin_state
        "Station_ID",              # redundant with origin_state
        "airport_name_clean",      # redundant with origin_state/lat-long
        "altitude_bin",            # redundant with continuous 'altitude'
    ]

    y = df[target_col].copy()
    X = df.drop(columns=drop_cols + [target_col])

    # Column groups for preprocessing
    categorical_cols = X.select_dtypes(include=["object", "bool", "str"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Cast categoricals to plain string dtype. Without this, mixing boolean
    # columns (pilot_warned, is_high_altitude) with text columns in the same
    # DataFrame slice confuses sklearn's dtype detection and it tries to
    # coerce everything to float, which crashes on the text columns.
    X[categorical_cols] = X[categorical_cols].astype(str)

    print(f"Features kept: {X.shape[1]}")
    print(f"  Numeric ({len(numeric_cols)}): {numeric_cols}")
    print(f"  Categorical ({len(categorical_cols)}): {categorical_cols}")

    return X, y, numeric_cols, categorical_cols


# ---------------------------------------------------------------------------
# 3. Shared preprocessing pipeline
# ---------------------------------------------------------------------------
def build_preprocessor(numeric_cols, categorical_cols):
    """
    One shared ColumnTransformer used by all three models so the comparison
    is apples-to-apples:
      - numeric columns  -> median-impute, then standardize (mean 0, std 1)
      - categorical cols -> most-frequent-impute, then one-hot encode

    Scaling matters a lot for SVM (distance-based), a little for Gaussian
    Naive Bayes, and not at all for Random Forest - but applying it uniformly
    doesn't hurt the tree model and keeps the pipeline identical across models.
    """
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # sparse_output=False: GaussianNB can't accept sparse matrices, and
        # the dataset is small enough that a dense matrix is no problem.
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])
    return preprocessor


# ---------------------------------------------------------------------------
# 4. Define the three models
# ---------------------------------------------------------------------------
def get_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=RANDOM_STATE,
        ),
    }


# ---------------------------------------------------------------------------
# 5. Train + evaluate each model
# ---------------------------------------------------------------------------
def run_experiment(X, y, numeric_cols, categorical_cols):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    models = get_models()

    results = {}
    fitted_pipelines = {}
    label_names = {0: "Low", 1: "Medium", 2: "High"}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", model),
        ])

        # 5-fold cross-validation on the training set (checks stability,
        # not just a single lucky/unlucky train-test split)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)

        # Fit on the full training set, evaluate on the held-out test set
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")

        print(f"\n{'='*60}\n{name}\n{'='*60}")
        print(f"5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"Test accuracy:      {acc:.3f}")
        print(f"Test macro F1:      {f1_macro:.3f}")
        print("\nClassification report (test set):")
        print(classification_report(y_test, y_pred, target_names=[label_names[c] for c in sorted(y.unique())]))

        results[name] = {
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "test_accuracy": acc,
            "test_f1_macro": f1_macro,
            "y_pred": y_pred,
        }
        fitted_pipelines[name] = pipe

    return results, fitted_pipelines, X_test, y_test, label_names


# ---------------------------------------------------------------------------
# 6. Visual comparison: confusion matrices + accuracy bar chart
# ---------------------------------------------------------------------------
def plot_results(results, y_test, label_names, out_path="model_comparison.png"):
    names = list(results.keys())
    fig, axes = plt.subplots(1, len(names) + 1, figsize=(6 * (len(names) + 1), 5))

    for ax, name in zip(axes[:-1], names):
        cm = confusion_matrix(y_test, results[name]["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=[label_names[c] for c in sorted(y_test.unique())])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name}\nAccuracy: {results[name]['test_accuracy']:.3f}")

    # Bar chart comparing test accuracy across models
    ax = axes[-1]
    accs = [results[n]["test_accuracy"] for n in names]
    bars = ax.bar(names, accs, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Model Comparison")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02, f"{acc:.3f}", ha="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved comparison chart to {out_path}")


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compare RF / Naive Bayes / SVM on bird strike risk data")
    parser.add_argument("--data", default="bird_collision_model_dataset_FINAL.csv", help="Path to the CSV file")
    parser.add_argument("--out", default="model_comparison.png", help="Path to save the comparison chart")
    args = parser.parse_args()

    df = load_data(args.data)
    X, y, numeric_cols, categorical_cols = prepare_features(df)
    results, pipelines, X_test, y_test, label_names = run_experiment(X, y, numeric_cols, categorical_cols)
    # Save each trained pipeline (preprocessing + model together) so the
    # UI/dashboard can load them later without retraining.
    for name, pipe in pipelines.items():
        filename = f"models/{name.replace(' ', '_').lower()}_model.pkl"
        with open(filename, "wb") as f:
            pickle.dump(pipe, f)
        print(f"Saved {name} pipeline to {filename}")
    plot_results(results, y_test, label_names, out_path=args.out)

    # Final summary table
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    summary = pd.DataFrame({
        name: {"CV Accuracy": r["cv_mean"], "Test Accuracy": r["test_accuracy"], "Test Macro F1": r["test_f1_macro"]}
        for name, r in results.items()
    }).T
    print(summary.round(3))


if __name__ == "__main__":
    main()
