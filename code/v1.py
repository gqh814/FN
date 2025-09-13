#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
df = pd.read_csv("/Users/theodorbock/Desktop/fndata.csv", header=1)

# Ensure the numeric columns are treated as numbers
years = [str(y) for y in range(2000, 2011)]
df[years] = df[years].apply(pd.to_numeric, errors="coerce")

# Drop rows with missing values in the critical years
df = df.dropna(subset=["2000", "2010"])

# Initial score (year 2000)
df["initial"] = df["2000"]

# Growth rate: (last - first) / first
df["growth_rate"] = (df["2010"] - df["2000"]) 

# Scatter plot
plt.figure(figsize=(10,6))
plt.scatter(df["initial"], df["growth_rate"], alpha=0.7)

# Add country labels
for i, row in df.iterrows():
    plt.text(row["initial"], row["growth_rate"], row["ISO"], fontsize=8)

# --- Linear fit ---
x = df["initial"]
y = df["growth_rate"]

m, b = np.polyfit(x, y, 1)  # slope, intercept
plt.plot(x, m*x + b, color="red", linewidth=2, label=f"Fit: y={m:.3f}x+{b:.3f}")

plt.xlabel("Initial Score (2000)")
plt.ylabel("Growth Rate (2000–2010)")
plt.title("Growth Rate vs Initial Score (2000–2010)")
plt.legend()
plt.grid(True)
plt.show()
#%%