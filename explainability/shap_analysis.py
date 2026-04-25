import shap
import pandas as pd
from utils import load_model

model = load_model("models/icu_model.pkl")

df = pd.read_csv("data/sepsis_icu.csv")
df = df.fillna(df.mean(numeric_only=True))

X = df.drop("SepsisLabel", axis=1)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X)