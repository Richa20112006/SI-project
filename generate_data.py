import numpy as np
import pandas as pd

np.random.seed(42)
n = 2000
df = pd.DataFrame({
    "city": np.random.choice(["Mumbai", "Delhi", "Bengaluru"], n),
    "room_type": np.random.choice(["Entire home", "Private room", "Shared room"], n, p=[0.55, 0.35, 0.10]),
    "accommodates": np.random.randint(1, 9, n),
    "number_of_reviews": np.random.poisson(25, n),
    "rating": np.random.normal(4.6, 0.3, n).clip(1, 5).round(2),
})
df["bedrooms"] = (df["accommodates"] / 2 + np.random.normal(0, 0.5, n)).round().clip(1, 6)
city_base = df["city"].map({"Mumbai": 3500, "Delhi": 2800, "Bengaluru": 2400})
room_mult = df["room_type"].map({"Entire home": 1.0, "Private room": 0.55, "Shared room": 0.3})
df["price"] = (city_base * room_mult * (1 + 0.12 * df["accommodates"]) * np.random.lognormal(0, 0.35, n)).round()
df.loc[np.random.choice(n, 80, replace=False), "rating"] = np.nan
df.loc[np.random.choice(n, 8, replace=False), "price"] *= 15
df["rating"] = df["rating"].fillna(df["rating"].median())
df = df[df["price"] < df["price"].quantile(0.99)]
df.to_csv("airbnb.csv", index=False)
print(f"Created airbnb.csv with {len(df)} rows.")
