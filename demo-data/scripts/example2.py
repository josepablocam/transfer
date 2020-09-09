import pandas as pd

df = pd.read_csv("data.csv")
df["a_str"] = df["a"].astype(str)
df = df[~pd.isnull(df["b"])]
other_df = df[df["a"] > 2]
other_df["a_twice"] = other_df["a"] * 2
