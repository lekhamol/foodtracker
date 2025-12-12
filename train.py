import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load SQL data exported as CSV
data = pd.read_csv("data.csv")  # <-- replace with your actual file

# Rename columns to a clean format (as seen in your screenshot)
data = data.rename(columns={
    "headcount": "Headcount",
    "food_prepared": "Food_Prepared",
    "food_consumed": "Food_Consumed",
    "food_wasted": "Food_Wasted"
})

print("Columns After Renaming:")
print(data.columns.tolist())

# Features (input)
X = data[["Headcount", "Food_Consumed", "Food_Wasted"]]

# Target (output)
y = data["Food_Prepared"]

# Train model
model = LinearRegression()
model.fit(X, y)

print("✔ Model Training Complete!")

# Save the trained model
pickle.dump(model, open("model.pkl", "wb"))

print("✔ model.pkl saved successfully!")
