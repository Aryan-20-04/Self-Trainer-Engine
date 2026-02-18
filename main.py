import pandas as pd
from core.engine import SelfTrainerEngine

df = pd.read_csv("data/telco_churn_data.csv")

# Drop ID
df = df.drop("customerID", axis=1)

# Fix target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Fix numeric column
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

engine = SelfTrainerEngine(mode="full")
engine.fit(df, target="Churn")

engine.summary()
engine.explain_global()
