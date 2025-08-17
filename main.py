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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"


def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline(
        [("Imputer", SimpleImputer(strategy="median")),
         ("Scaler", StandardScaler())]
    )

    cat_pipeline = Pipeline(
        [("Imputer", SimpleImputer(strategy="most_frequent")),
         ("OneHot", OneHotEncoder(handle_unknown="ignore"))]
    )

    full_pipeline = ColumnTransformer(
        [("num", num_pipeline, num_attribs),
         ("cat", cat_pipeline, cat_attribs)]
    )

    return full_pipeline


# ======================================
# Training Part
# ======================================
if not os.path.exists(MODEL_FILE):

    heart = pd.read_csv("heart.csv")

    # Create stratified split based on age group
    heart["age_grp"] = pd.cut(
        heart["age"], bins=[0, 20, 40, 60, np.inf], labels=[1, 2, 3, 4]
    )
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(heart, heart["age_grp"]):
        train_set = heart.loc[train_index].drop("age_grp", axis=1)
        test_set = heart.loc[test_index].drop("age_grp", axis=1)

    # Save test set for inference
    test_set.to_csv("input.csv", index=False)

    # Split features and labels
    train_labels = train_set["target"]
    train_features = train_set.drop("target", axis=1)

    test_labels = test_set["target"]
    test_features = test_set.drop("target", axis=1)

    # Select numerical & categorical attributes
    num_attribs = train_features.select_dtypes(include=[np.number]).columns.tolist()
    cat_attribs = train_features.select_dtypes(exclude=[np.number]).columns.tolist()

    # Build pipeline
    pipeline = build_pipeline(num_attribs, cat_attribs)
    train_prepared = pipeline.fit_transform(train_features)
    test_prepared = pipeline.transform(test_features)

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(train_prepared, train_labels)

    # Save model & pipeline
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("✅ Model trained and saved!")

    # Evaluate accuracy on test set
    test_preds = model.predict(test_prepared)
    acc = accuracy_score(test_labels, test_preds)
    print("🎯 Accuracy on test set:", acc)

    # Cross-validation
    scores = cross_val_score(model, train_prepared, train_labels, cv=5)
    print("📊 Cross-validation scores:", scores)
    print("🔑 Average accuracy:", scores.mean())

    # Confusion Matrix 
    cm = confusion_matrix(test_labels, test_preds)
    print("Confusion Matrix:\n", cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    # Feature Importance
    importances = model.feature_importances_
    feature_names = pipeline.get_feature_names_out()
    plt.barh(feature_names, importances)
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Feature Importance from RandomForest")
    plt.show()

    # Save test predictions
    test_set["predictions"] = test_preds
    test_set.to_csv("output.csv", index=False)
    print("📂 Predictions with target saved to output.csv")

# ======================================
# Inference Part (if model already exists)
# ======================================
else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("input.csv")

    X_input = input_data.drop(columns=["target"], errors="ignore")
    y_true = input_data["target"] if "target" in input_data.columns else None

    X_prepared = pipeline.transform(X_input)
    predictions = model.predict(X_prepared)

    input_data["predictions"] = predictions
    input_data.to_csv("output.csv", index=False)

    print("✅ Inference complete! Results saved to output.csv")

    # Confusion Matrix
    if y_true is not None:
        cm = confusion_matrix(y_true, predictions)
        print("Confusion Matrix:\n", cm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()
