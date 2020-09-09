import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("data.csv")
df["a"] = df["a"] * 2

summary = df.groupby("a")["b"].sum()
summary = summary.to_frame(name="b_sum").reset_index()

fig, ax = plt.subplots(1)
ax.plot(summary["a"], summary["b_sum"])


other_df = df.groupby("a")[["b", "c"]].sum().reset_index()
other_df = other_df.rename(columns={"b": "b_sum", "c": "c_sum"})

combined_df = pd.merge(df, other_df, how="left", on="a")
combined_df["final_a"] = combined_df["a"] + combined_df["b_sum"] + combined_df["c_sum"]
