import pandas as pd
import mysql.connector
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle

# ===========================================
# 1. CONNECT TO MYSQL
# ===========================================
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root123",
    database="foodtracker"
)

# ===========================================
# 2. LOAD DATA
# ===========================================
query = "SELECT * FROM food_data"
data = pd.read_sql(query, db)

data.columns = data.columns.str.strip()

print("\n📌 Data Loaded from SQL:")
print(data.head())

# ===========================================
# 3. COLUMN STANDARDIZATION
# ===========================================
column_map = {
    'headcount': 'Headcount',
    'persons': 'Headcount',
    'count': 'Headcount',

    'consumed': 'Food_Consumed',
    'food_consumed': 'Food_Consumed',

    'waste': 'Food_Wasted',
    'food_wasted': 'Food_Wasted',

    'prepared': 'Food_Prepared',
    'food_prepared': 'Food_Prepared'
}

data = data.rename(columns={old: new for old, new in column_map.items() if old in data.columns})

# ===========================================
# 4. REQUIRED COLUMN CHECK
# ===========================================
required_cols = ['Headcount', 'Food_Consumed', 'Food_Wasted', 'Food_Prepared']
for col in required_cols:
    if col not in data.columns:
        raise ValueError(f"❌ Missing column: {col}")

# ===========================================
# 5. TRAINING DATA
# ===========================================
X = data[['Headcount', 'Food_Consumed', 'Food_Wasted']]
y = data['Food_Prepared']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===========================================
# 6. FEATURE SCALING (VERY IMPORTANT)
# ===========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("\n✅ Model Training Complete!")

# ===========================================
# 7. USER INPUT
# ===========================================
print("\n---------------------------------------------")
print("🍽 FOOD PREPARATION PREDICTION SYSTEM")
print("---------------------------------------------")

headcount = int(input("\nEnter expected headcount for tomorrow: "))
consumed = float(input("Enter today's food consumed (kg): "))
wasted = float(input("Enter today's food wasted (kg): "))

# ===========================================
# 8. CORRECT PREDICTION METHOD
# ===========================================
input_data = pd.DataFrame([{
    "Headcount": headcount,
    "Food_Consumed": consumed,
    "Food_Wasted": wasted
}])

input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)

# Prevent negative food values
final_prediction = max(0, prediction[0])

print(f"\n🍽 Recommended food to prepare for tomorrow: {final_prediction:.2f} kg")

# ===========================================
# 9. SAVE MODEL & SCALER
# ===========================================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
