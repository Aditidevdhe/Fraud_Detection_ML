# train_and_save.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# Load your dataset
df = pd.read_csv("AIML Dataset.csv")

# Ensure these columns exist in your dataset
X = df[["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]]
y = df["isFraud"]  # Change this if your fraud label column has a different name

# Define preprocessing
preprocessor = ColumnTransformer([
    ("type", OneHotEncoder(handle_unknown="ignore"), ["type"]),
    ("num", StandardScaler(), ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"])
])

# Build the pipeline
pipe = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier())
])

# Train and save the model
pipe.fit(X, y)
joblib.dump(pipe, "fraud_detection_pipeline.pkl")
