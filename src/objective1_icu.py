import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from utils import save_model
import os

# ----------------------------
# FILE PATH
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(BASE_DIR, "data", "sepsis_icu_synthetic.csv")

# Load dataset
df = pd.read_csv(file_path)

print("\nDataset loaded successfully")
print("Columns:", df.columns)

# ----------------------------
# HANDLE MISSING VALUES
# ----------------------------
df = df.fillna(df.mean(numeric_only=True))

# ----------------------------
# TARGET COLUMN
# ----------------------------
target = "sepsis_label"

print("\nUsing target column:", target)

# ----------------------------
# ENCODE CATEGORICAL DATA
# ----------------------------
label_encoders = {}

for col in df.columns:
    if df[col].dtype == "object" and col != target:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# ----------------------------
# SPLIT DATA
# ----------------------------
X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ----------------------------
# MODEL TRAINING
# ----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------
# PREDICTION
# ----------------------------
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("\nICU Model Accuracy:", acc)

# ----------------------------
# FEATURE IMPORTANCE (VIVA POINT)
# ----------------------------
print("\nTop Features:")
importances = model.feature_importances_
features = X.columns

for i in sorted(zip(importances, features), reverse=True)[:10]:
    print(i)

# ----------------------------
# SAVE MODEL
# ----------------------------
save_model(model, os.path.join(BASE_DIR, "models", "icu_model.pkl"))

print("\n✅ ICU model saved successfully")