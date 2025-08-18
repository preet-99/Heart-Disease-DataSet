import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
    classification_report
)

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

# ==============================
# Function to build preprocessing pipeline
# ==============================
def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])

    return full_pipeline


# ==============================
# Training
# ==============================
def train_model():
    heart = pd.read_csv("heart.csv")

    # Stratified split based on descriptive age groups
    heart["age_grp"] = pd.cut(
        heart["age"], bins=[0, 20, 40, 60, np.inf],
        labels=["teen", "young", "middle", "senior"]
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in split.split(heart, heart["age_grp"]):
        train_set = heart.loc[train_idx].drop("age_grp", axis=1)
        test_set = heart.loc[test_idx].drop("age_grp", axis=1)

    # Save test set for inference
    test_set.to_csv("input.csv", index=False)

    # Split features and labels
    X_train_features = train_set.drop("target", axis=1)
    y_train_label = train_set["target"]

    X_test_features = test_set.drop("target", axis=1)
    y_test_label = test_set["target"]

    # Numerical & categorical attributes
    num_attribs = X_train_features.select_dtypes(include=[np.number]).columns.tolist()
    cat_attribs = X_train_features.select_dtypes(exclude=[np.number]).columns.tolist()

    # Build pipeline
    pipeline = build_pipeline(num_attribs, cat_attribs)
    X_train_prepared = pipeline.fit_transform(X_train_features)
    X_test_prepared = pipeline.transform(X_test_features)

    # Full pipeline with model for easier cross-validation
    full_model = Pipeline([
        ("preprocess", pipeline),
        ("clf", RandomForestClassifier(random_state=42))
    ])

    # Train model
    full_model.fit(X_train_features, y_train_label)
    joblib.dump(full_model, MODEL_FILE)
    print("✅ Model trained and saved!")

    # Evaluation
    y_pred = full_model.predict(X_test_features)
    print("🎯 Accuracy on test set:", accuracy_score(y_test_label, y_pred))
    print("📄 Classification Report:\n", classification_report(y_test_label, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test_label, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    # Cross-validation
    scores = cross_val_score(full_model, X_train_features, y_train_label, cv=5)
    print("📊 Cross-validation scores:", scores)
    print("🔑 Average CV accuracy:", scores.mean())

    # Feature Importance
    model = full_model.named_steps["clf"]
    feature_names = full_model.named_steps["preprocess"].get_feature_names_out()
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 10))
    plt.barh(np.array(feature_names)[indices], importances[indices])
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("RandomForest Feature Importance")
    plt.show()

    # Save test predictions
    test_set["predictions"] = y_pred
    test_set.to_csv("output.csv", index=False)
    print("📂 Predictions saved to output.csv")


# ==============================
# Inference
# ==============================
def inference():
    if not os.path.exists(MODEL_FILE):
        print("❌ Model not found. Train the model first.")
        return

    full_model = joblib.load(MODEL_FILE)
    input_data = pd.read_csv("input.csv")

    X_input = input_data.drop(columns=["target"], errors="ignore")
    y_true = input_data["target"] if "target" in input_data.columns else None

    predictions = full_model.predict(X_input)
    input_data["predictions"] = predictions
    input_data.to_csv("output.csv", index=False)
    print("✅ Inference complete! Results saved to output.csv")

    if y_true is not None:
        print("📄 Classification Report:\n", classification_report(y_true, predictions))
        cm = confusion_matrix(y_true, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()


# ==============================
# Main
# ==============================
if __name__ == "__main__":
    if not os.path.exists(MODEL_FILE):
        train_model()
    else:
        inference()
