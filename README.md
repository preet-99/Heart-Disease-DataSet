# 🫀 Heart Disease Prediction using Random Forest

This project implements a **machine learning pipeline** to predict heart disease using the **UCI Heart Disease dataset**.  
The pipeline includes data preprocessing, feature scaling, categorical encoding, and training a **RandomForestClassifier** model.

---

## 📂 Project Structure

```text
├── heart.csv          # Dataset file (input data)  
├── model.pkl          # Trained Random Forest model (saved after training)  
├── pipeline.pkl       # Preprocessing pipeline (saved after training)  
├── input.csv          # Test set used for inference  
├── output.csv         # Predictions with target values  
├── main.py            # Main Python script (entry point)  
├── requirements.txt   # Project dependencies  
└── README.md          # Project documentation  


---

## ⚙️ Features

- Preprocessing pipeline:
  - Imputation of missing values
  - Standard scaling for numerical features
  - One-hot encoding for categorical features
  - Train/test split using **StratifiedShuffleSplit**
  - Random Forest classifier
  - Model persistence using `joblib`
  - Evaluation:
  - Accuracy on test set
  - Cross-validation scores
  - **Feature importance visualization**

---

## 🛠️ Installation

Clone the repository and install dependencies:

``` bash
git clone https://github.com/preet-99/Heart-Disease-DataSet.git
cd heart-disease-prediction
pip install -r requirements.txt
