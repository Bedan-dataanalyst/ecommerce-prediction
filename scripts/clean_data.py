import pandas as pd

df = pd.read_csv("data/Online_retail.csv", encoding="latin1")

print("Original shape:", df.shape)

df = df.dropna(subset=["CustomerID"])
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]

df["Revenue"] = df["Quantity"] * df["UnitPrice"]

print("Cleaned shape:", df.shape)

df.to_csv("data/cleaned_retail.csv", index=False)

print("Saved cleaned_retail.csv")