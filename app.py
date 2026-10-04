from flask import Flask, render_template, request
import pandas as pd
import joblib
import json

from recommendation import recommend_from_user_input


# ==========================================================
# CREATE FLASK APPLICATION
# ==========================================================

app = Flask(__name__)


# ==========================================================
# LOAD TRAINED PRICE MODEL
# ==========================================================

model = joblib.load("models/price_model.pkl")


# ==========================================================
# LOAD METADATA
# ==========================================================

with open("models/metadata.json", "r") as file:
    metadata = json.load(file)


# ==========================================================
# HOME PAGE
# ==========================================================

@app.route("/")
def home():

    return render_template(

        "index.html",

        brands=metadata["brand"],
        brand_models=metadata["brand_models"],
        fuels=metadata["fuel"],
        seller_types=metadata["seller_type"],
        transmissions=metadata["transmission"],
        owners=metadata["owner"]

    )


# ==========================================================
# PRICE PREDICTION + RECOMMENDATION
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # --------------------------------------------------
        # 1. GET USER INPUT
        # --------------------------------------------------

        brand = request.form["brand"]

        vehicle_model = request.form["model"]

        year = int(
            request.form["year"]
        )

        km_driven = int(
            request.form["km_driven"]
        )

        fuel = request.form["fuel"]

        seller_type = request.form["seller_type"]

        transmission = request.form["transmission"]

        owner = request.form["owner"]

        mileage = float(
            request.form["mileage"]
        )

        engine = float(
            request.form["engine"]
        )

        seats = float(
            request.form["seats"]
        )


        # --------------------------------------------------
        # 2. PREPARE DATA FOR PRICE MODEL
        # --------------------------------------------------

        input_data = pd.DataFrame({

            "brand": [brand],

            "model": [vehicle_model],

            "year": [year],

            "km_driven": [km_driven],

            "fuel": [fuel],

            "seller_type": [seller_type],

            "transmission": [transmission],

            "owner": [owner],

            "mileage_num": [mileage],

            "engine_num": [engine],

            "seats": [seats]

        })


        # --------------------------------------------------
        # 3. PREDICT PRICE
        # --------------------------------------------------

        prediction = model.predict(
            input_data
        )[0]


        prediction = max(
            0,
            prediction
        )


        predicted_price = "₹{:,.0f}".format(
            prediction
        )


        # --------------------------------------------------
        # 4. GET SIMILAR VEHICLES
        # --------------------------------------------------

        recommendations = recommend_from_user_input(

    brand=brand,

    model=vehicle_model,

    year=year,

    predicted_price=prediction,

    km_driven=km_driven,

    fuel=fuel,

    seller_type=seller_type,

    transmission=transmission,

    owner=owner,

    mileage=mileage,

    engine=engine,

    seats=seats,

    number_of_similar=3,

    number_of_alternatives=3

)


        # --------------------------------------------------
        # 5. RETURN RESULTS
        # --------------------------------------------------

        return render_template(

            "index.html",

            prediction=predicted_price,

            recommendations=recommendations,

            brands=metadata["brand"],
            brand_models=metadata["brand_models"],
            fuels=metadata["fuel"],
            seller_types=metadata["seller_type"],
            transmissions=metadata["transmission"],
            owners=metadata["owner"],

            selected_brand=brand,
            selected_model=vehicle_model

        )


    except Exception as error:

        print(
            "ERROR:",
            error
        )

        return render_template(

            "index.html",

            error=str(error),

            brands=metadata["brand"],
            brand_models=metadata["brand_models"],
            fuels=metadata["fuel"],
            seller_types=metadata["seller_type"],
            transmissions=metadata["transmission"],
            owners=metadata["owner"]

        )


# ==========================================================
# RUN FLASK
# ==========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )