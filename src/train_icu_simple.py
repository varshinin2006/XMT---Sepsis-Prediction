import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ----------------------------
# LOAD DATA
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

df = pd.read_csv(os.path.join(BASE_DIR, "data/sepsis_icu_synthetic.csv"))

# ----------------------------
# SELECT ONLY IMPORTANT FEATURES
# ----------------------------
features = [
    "age",
    "bmi",
    "lactate_mmol",
    "wbc",
    "sofa_score",
    "creatinine",
    "gender"
]

target = "sepsis_label"

# ----------------------------
# CLEAN DATA
# ----------------------------
df = df.fillna(df.mean(numeric_only=True))

# Convert gender → numeric
df["gender"] = df["gender"].map({"M": 1, "F": 0, "Male":1, "Female":0})

# ----------------------------
# FINAL DATA
# ----------------------------
X = df[features]
y = df[target]

# ----------------------------
# TRAIN TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ----------------------------
# MODEL
# ----------------------------
model = RandomForestClassifier(class_weight="balanced")
model.fit(X_train, y_train)

# ----------------------------
# SAVE MODEL + FEATURES
# ----------------------------
joblib.dump(model, os.path.join(BASE_DIR, "models/icu_model.pkl"))
joblib.dump(features, os.path.join(BASE_DIR, "models/icu_features.pkl"))

print("✅ Model retrained successfully!")