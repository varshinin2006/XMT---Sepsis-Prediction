import pandas as pd
from utils import load_model
import os
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# PATH SETUP
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

model = load_model(os.path.join(BASE_DIR, "models", "icu_model.pkl"))

df = pd.read_csv(os.path.join(BASE_DIR, "data", "sepsis_icu_synthetic.csv"))

target = "sepsis_label"

print("\nDataset loaded for testing")

# ----------------------------
# HANDLE MISSING VALUES
# ----------------------------
df = df.fillna(df.mean(numeric_only=True))

# ----------------------------
# ENCODE CATEGORICAL FEATURES
# ----------------------------
for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# ----------------------------
# FEATURES
# ----------------------------
X = df.drop(target, axis=1)

# ----------------------------
# TEST MULTIPLE SAMPLES (IMPORTANT FOR DEMO)
# ----------------------------
print("\n--- PREDICTION RESULTS ---")

for i in range(5):
    sample = X.iloc[[i]]

    # Prediction class
    pred = model.predict(sample)[0]

    # Probability (IMPORTANT FOR VIVA)
    prob = model.predict_proba(sample)[0][1]

    print(f"\nPatient {i+1}:")
    print("Sepsis Probability:", round(prob, 4))

    if pred == 1:
        print("🚨 RESULT: SEPSIS RISK")
    else:
        print("✅ RESULT: NO SEPSIS RISK")