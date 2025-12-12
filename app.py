from flask import Flask, render_template, request
import numpy as np
import pickle
import mysql.connector

app = Flask(__name__)

# --------------------------------------
# Load Model
# --------------------------------------
model = pickle.load(open("model.pkl", "rb"))

# --------------------------------------
# MySQL Connection (optional)
# --------------------------------------
def get_db():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="food_predictor"
        )
        return conn
    except:
        return None

# --------------------------------------
# Home Route
# --------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# --------------------------------------
# Predict Route (PASTE HERE!)
# --------------------------------------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            headcount = int(request.form["headcount"])
            consumed = float(request.form["consumed"])
            wasted = float(request.form["wasted"])

            # Input for model
            inp = np.array([[headcount, consumed, wasted]])
            output = model.predict(inp)[0]

            # Save into MySQL (optional)
            conn = get_db()
            if conn:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO prediction_data (headcount, consumed, wasted, predicted) VALUES (%s, %s, %s, %s)",
                    (headcount, consumed, wasted, output)
                )
                conn.commit()

            return render_template(
                "index.html",
                prediction=f"You should prepare {output:.2f} units of food"
            )

        except Exception as e:
            return render_template("index.html", prediction=f"Error: {e}")

    # GET request (prevents 405 error)
    return render_template("index.html")

# --------------------------------------
# Run App
# --------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
