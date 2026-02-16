import pandas as pd
from core.engine import SelfTrainerEngine

df = pd.read_csv("data/creditcard.csv")
df = df.sample(frac=0.3, random_state=42)

engine = SelfTrainerEngine()
engine.fit(df, target="Class")

engine.summary()

engine.explain_global()

# Explain one sample
X_sample = df.drop("Class", axis=1).sample(1, random_state=42)
engine.explain_instance(X_sample)
