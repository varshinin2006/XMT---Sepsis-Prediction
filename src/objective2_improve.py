import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/sepsis_icu.csv")
df = df.fillna(df.mean(numeric_only=True))

target = "SepsisLabel"

X = df.drop(target, axis=1)
y = df[target]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

model = GradientBoostingClassifier()
model.fit(X_train, y_train)

print("Improved Model Accuracy:", model.score(X_test, y_test))