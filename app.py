from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

# Load Model, Scaler and Columns
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")   # list of columns used in training


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        try:
            # Get values from form
            area = float(request.form["area"])
            bedrooms = int(request.form["bedrooms"])
            bathrooms = int(request.form["bathrooms"])
            # stories = int(request.form["stories"])
            parking = int(request.form["parking"])

            # mainroad = request.form["mainroad"]
            # guestroom = request.form["guestroom"]
            # basement = request.form["basement"]
            # hotwaterheating = request.form["hotwaterheating"]
            airconditioning = request.form["airconditioning"]
            # prefarea = request.form["prefarea"]

            furnishingstatus = request.form["furnishingstatus"]

            # Convert yes/no to 1/0
            def yes_no(value):
                return 1 if value == "yes" else 0

            # Base input dictionary
            input_dict = {
                "area": area,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                # "stories": stories,
                # "mainroad": yes_no(mainroad),
                # "guestroom": yes_no(guestroom),
                # "basement": yes_no(basement),
                # "hotwaterheating": yes_no(hotwaterheating),
                "airconditioning": yes_no(airconditioning),
                "parking": parking,
                # "prefarea": yes_no(prefarea),
                "furnishingstatus_semi-furnished": 0,
                "furnishingstatus_unfurnished": 0
            }

            # Furnishing status one-hot encoding
            if furnishingstatus == "semi-furnished":
                input_dict["furnishingstatus_semi-furnished"] = 1
            elif furnishingstatus == "unfurnished":
                input_dict["furnishingstatus_unfurnished"] = 1

            # Convert to DataFrame
            input_df = pd.DataFrame([input_dict])

            # Ensure correct column order
            input_df = input_df.reindex(columns=columns, fill_value=0)

            # Scale
            input_scaled = scaler.transform(input_df)

            # Predict
            result = model.predict(input_scaled)[0]

            prediction = round(result, 2)

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
