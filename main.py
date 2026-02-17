import pandas as pd
from core.engine import SelfTrainerEngine

df = pd.read_csv("data/creditcard.csv")
df = df.sample(frac=0.3, random_state=42)

engine = SelfTrainerEngine()
engine.fit(df, target="Class")

engine.summary()

engine.explain_global()
drifted_df = df.drop("Class", axis=1).copy()
drifted_df["V14"] *= 1.5

engine.check_drift(drifted_df)

# Explain one sample
X_sample = df.drop("Class", axis=1).sample(1, random_state=42)
engine.explain_instance(X_sample)

# -----------------------------
# Test Model Loading
# -----------------------------
print("\nTesting model reload...")

new_engine = SelfTrainerEngine()
new_engine.load(engine.model_path, engine.meta_path)

# Test prediction after loading
prediction = new_engine.predict(X_sample)
print("Prediction after loading:", prediction)