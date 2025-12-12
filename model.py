import pandas as pd
import mysql.connector
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ===========================================
# 1. CONNECT TO MYSQL
# ===========================================
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root123",     # 🔴 CHANGE THIS
    database="foodtracker"      # 🔴 CHANGE THIS
)

# ===========================================
# 2. LOAD DATA FROM SQL TABLE
# ===========================================
query = "SELECT * FROM food_data"   # 🔴 CHANGE TABLE NAME
data = pd.read_sql(query, db)

# Clean column names
data.columns = data.columns.str.strip()

print("\n📌 Data Loaded from SQL:")
print(data.head())

print("\nAvailable Columns in SQL Table:")
print(list(data.columns))

# ===========================================
# 3. OPTIONAL COLUMN AUTO-RENAMING
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

print("\nColumns After Renaming:")
print(list(data.columns))

# ===========================================
# 4. VALIDATION OF REQUIRED COLUMNS
# ===========================================
required_cols = ['Headcount', 'Food_Consumed', 'Food_Wasted', 'Food_Prepared']

for col in required_cols:
    if col not in data.columns:
        raise ValueError(f"❌ ERROR: Missing column '{col}' in SQL table.")

# ===========================================
# 5. MODEL TRAINING
# ===========================================
X = data[['Headcount', 'Food_Consumed', 'Food_Wasted']]
y = data['Food_Prepared']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("\n✅ Model Training Complete!")

# ===========================================
# 6. PREDICTION INPUT
# ===========================================
print("\n---------------------------------------------")
print("🍽 FOOD PREPARATION PREDICTION SYSTEM")
print("---------------------------------------------")

headcount = int(input("\nEnter expected headcount for tomorrow: "))
consumed = float(input("Enter today's food consumed (kg): "))
wasted = float(input("Enter today's food wasted (kg): "))

# ===========================================
# 7. PREDICTION OUTPUT
# ===========================================
prediction = model.predict([[headcount, consumed, wasted]])

print(f"\n🍽 Recommended food to prepare for tomorrow: {prediction[0]:.2f} kg")
import pickle
pickle.dump(model, open("model.pkl", "wb"))

