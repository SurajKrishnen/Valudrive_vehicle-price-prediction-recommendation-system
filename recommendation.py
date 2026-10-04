import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================================
# 1. LOAD DATASET
# ==========================================================

df = pd.read_csv("data/Car details v3.csv")

# Remove duplicate records
df = df.drop_duplicates().reset_index(drop=True)


# ==========================================================
# 2. EXTRACT BRAND AND MODEL
# ==========================================================

# Example:
# "Maruti Baleno Delta 1.2"
# Brand = Maruti
# Model = Baleno

df["brand"] = (
    df["name"]
    .astype(str)
    .str.split()
    .str[0]
)

df["model"] = (
    df["name"]
    .astype(str)
    .str.split()
    .str[1]
)


# ==========================================================
# 3. CLEAN NUMERICAL COLUMNS
# ==========================================================

# Mileage:
# "21.4 kmpl" -> 21.4

df["mileage_clean"] = pd.to_numeric(
    df["mileage"]
    .astype(str)
    .str.extract(r"([\d.]+)")[0],
    errors="coerce"
)


# Engine:
# "1197 CC" -> 1197

df["engine_clean"] = pd.to_numeric(
    df["engine"]
    .astype(str)
    .str.extract(r"([\d.]+)")[0],
    errors="coerce"
)


# Seats

df["seats"] = pd.to_numeric(
    df["seats"],
    errors="coerce"
)


# ==========================================================
# 4. CORRECT OBVIOUS FUEL INCONSISTENCIES
# ==========================================================

def correct_fuel_from_name(row):

    vehicle_name = str(row["name"]).upper()
    current_fuel = str(row["fuel"]).strip()

    # Only override when the vehicle name explicitly
    # identifies an alternative fuel type.

    if "CNG" in vehicle_name:
        return "CNG"

    if "LPG" in vehicle_name:
        return "LPG"

    return current_fuel


df["fuel"] = df.apply(
    correct_fuel_from_name,
    axis=1
)


# ==========================================================
# 5. HANDLE MISSING NUMERICAL VALUES
# ==========================================================

for column in [
    "mileage_clean",
    "engine_clean",
    "seats"
]:

    df[column] = df[column].fillna(
        df[column].median()
    )


# ==========================================================
# 6. NUMERICAL SIMILARITY FUNCTION
# ==========================================================

def numerical_similarity(
    value1,
    value2,
    tolerance
):

    try:

        difference = abs(
            float(value1)
            -
            float(value2)
        )

        score = 1 - (
            difference / tolerance
        )

        return max(
            0,
            min(1, score)
        )

    except (ValueError, TypeError):

        return 0


# ==========================================================
# 7. WEIGHTED MATCH SCORE
# ==========================================================

def calculate_match_score(

    car,

    brand,
    model,
    year,
    predicted_price,
    km_driven,
    fuel,
    seller_type,
    transmission,
    owner,
    mileage,
    engine,
    seats,

    same_model=False

):

    score = 0


    # ======================================================
    # BRAND + MODEL
    # ======================================================

    if same_model:

        # Strong weighting for actual same-model listings.

        if car["brand"] == brand:
            score += 15

        if car["model"] == model:
            score += 25

    else:

        # Alternatives receive only a small same-brand bonus.
        # This allows other brands to compete fairly.

        if car["brand"] == brand:
            score += 5


    # ======================================================
    # FUEL TYPE
    # ======================================================

    if car["fuel"] == fuel:
        score += 10


    # ======================================================
    # TRANSMISSION
    # ======================================================

    if car["transmission"] == transmission:
        score += 10


    # ======================================================
    # OWNERSHIP
    # ======================================================

    if car["owner"] == owner:
        score += 5


    # ======================================================
    # SELLER TYPE
    # ======================================================

    if car["seller_type"] == seller_type:
        score += 3


    # ======================================================
    # YEAR - MAXIMUM 10 POINTS
    # ======================================================

    score += (
        numerical_similarity(
            car["year"],
            year,
            5
        )
        * 10
    )


    # ======================================================
    # PRICE - MAXIMUM 10 POINTS
    # ======================================================

    score += (
        numerical_similarity(
            car["selling_price"],
            predicted_price,
            400000
        )
        * 10
    )


    # ======================================================
    # KM DRIVEN - MAXIMUM 7 POINTS
    # ======================================================

    score += (
        numerical_similarity(
            car["km_driven"],
            km_driven,
            60000
        )
        * 7
    )


    # ======================================================
    # ENGINE - MAXIMUM 3 POINTS
    # ======================================================

    score += (
        numerical_similarity(
            car["engine_clean"],
            engine,
            1000
        )
        * 3
    )


    # ======================================================
    # MILEAGE - MAXIMUM 1 POINT
    # ======================================================

    score += (
        numerical_similarity(
            car["mileage_clean"],
            mileage,
            15
        )
        * 1
    )


    # ======================================================
    # SEATS - MAXIMUM 1 POINT
    # ======================================================

    if car["seats"] == seats:
        score += 1


    return round(
        score,
        2
    )


# ==========================================================
# 8. COSINE SIMILARITY
# ==========================================================

def calculate_cosine_scores(

    candidate_df,

    year,
    predicted_price,
    km_driven,
    mileage,
    engine,
    seats

):

    if candidate_df.empty:
        return np.array([])


    numerical_columns = [
        "year",
        "selling_price",
        "km_driven",
        "mileage_clean",
        "engine_clean",
        "seats"
    ]


    candidate_values = (
        candidate_df[
            numerical_columns
        ]
        .copy()
        .astype(float)
    )


    # User vehicle represented using the same
    # numerical feature structure.

    user_values = pd.DataFrame({

        "year":
            [year],

        "selling_price":
            [predicted_price],

        "km_driven":
            [km_driven],

        "mileage_clean":
            [mileage],

        "engine_clean":
            [engine],

        "seats":
            [seats]

    })


    combined = pd.concat(
        [
            candidate_values,
            user_values
        ],
        ignore_index=True
    )


    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(
        combined
    )


    candidate_scaled = scaled[:-1]

    user_scaled = (
        scaled[-1]
        .reshape(1, -1)
    )


    similarities = cosine_similarity(
        user_scaled,
        candidate_scaled
    )[0]


    return similarities


# ==========================================================
# 9. BUILD EXPLAINABLE RESULT
# ==========================================================

def create_result(

    car,

    user_year=None,
    predicted_price=None,
    user_km=None,
    user_fuel=None,
    user_transmission=None,
    user_engine=None,
    user_seats=None

):

    reasons = []


    # ------------------------------------------------------
    # Fuel
    # ------------------------------------------------------

    if (
        user_fuel is not None
        and car["fuel"] == user_fuel
    ):

        reasons.append(
            "Same fuel type"
        )


    # ------------------------------------------------------
    # Transmission
    # ------------------------------------------------------

    if (
        user_transmission is not None
        and car["transmission"] == user_transmission
    ):

        reasons.append(
            "Same transmission"
        )


    # ------------------------------------------------------
    # Manufacturing year
    # ------------------------------------------------------

    if user_year is not None:

        year_difference = abs(
            int(car["year"])
            -
            int(user_year)
        )

        if year_difference == 0:

            reasons.append(
                "Same manufacturing year"
            )

        elif year_difference == 1:

            reasons.append(
                "Only 1 year difference"
            )

        elif year_difference <= 3:

            reasons.append(
                f"Only {year_difference} years difference"
            )


    # ------------------------------------------------------
    # Price
    # ------------------------------------------------------

    if predicted_price is not None:

        price_difference = abs(
            float(car["selling_price"])
            -
            float(predicted_price)
        )

        if price_difference <= 50000:

            reasons.append(
                "Very close price range"
            )

        elif price_difference <= 150000:

            reasons.append(
                "Similar price range"
            )


    # ------------------------------------------------------
    # KM driven
    # ------------------------------------------------------

    if user_km is not None:

        km_difference = abs(
            float(car["km_driven"])
            -
            float(user_km)
        )

        if km_difference <= 15000:

            reasons.append(
                "Similar kilometres driven"
            )


    # ------------------------------------------------------
    # Engine
    # ------------------------------------------------------

    if user_engine is not None:

        engine_difference = abs(
            float(car["engine_clean"])
            -
            float(user_engine)
        )

        if engine_difference <= 200:

            reasons.append(
                "Similar engine capacity"
            )


    # ------------------------------------------------------
    # Seats
    # ------------------------------------------------------

    if user_seats is not None:

        if (
            float(car["seats"])
            ==
            float(user_seats)
        ):

            reasons.append(
                "Same seating capacity"
            )


    # Keep the interface concise
    reasons = reasons[:4]


    return {

        "name":
            car["name"],

        "brand":
            car["brand"],

        "model":
            car["model"],

        "year":
            int(car["year"]),

        "price":
            int(car["selling_price"]),

        "km_driven":
            int(car["km_driven"]),

        "fuel":
            car["fuel"],

        "transmission":
            car["transmission"],

        "owner":
            car["owner"],

        "seller_type":
            car["seller_type"],

        "mileage":
            car["mileage"],

        "engine":
            car["engine"],

        "seats":
            (
                int(car["seats"])
                if pd.notna(car["seats"])
                else None
            ),

        "match_score":
            round(
                float(
                    car["final_score"]
                ),
                1
            ),

        "reasons":
            reasons
    }


# ==========================================================
# 10. MAIN RECOMMENDATION FUNCTION
# ==========================================================

def recommend_from_user_input(

    brand,
    model,
    year,
    predicted_price,
    km_driven,
    fuel,
    seller_type,
    transmission,
    owner,
    mileage,
    engine,
    seats,

    number_of_similar=3,
    number_of_alternatives=3

):


    # ======================================================
    # PART A - SAME MODEL LISTINGS
    # ======================================================

    same_model_df = df[

        (df["brand"] == brand)

        &

        (df["model"] == model)

    ].copy()


    # Prefer same-model vehicles reasonably close in age.

    nearby_same_model = same_model_df[

        abs(
            same_model_df["year"]
            -
            year
        )

        <= 5

    ].copy()


    if (
        len(nearby_same_model)
        >= number_of_similar
    ):

        same_model_df = nearby_same_model


    # ------------------------------------------------------
    # Weighted scores
    # ------------------------------------------------------

    same_scores = []


    for _, car in same_model_df.iterrows():

        score = calculate_match_score(

            car=car,

            brand=brand,
            model=model,
            year=year,
            predicted_price=predicted_price,
            km_driven=km_driven,
            fuel=fuel,
            seller_type=seller_type,
            transmission=transmission,
            owner=owner,
            mileage=mileage,
            engine=engine,
            seats=seats,

            same_model=True

        )

        same_scores.append(
            score
        )


    # ------------------------------------------------------
    # Cosine + weighted hybrid score
    # ------------------------------------------------------

    if not same_model_df.empty:

        same_model_df[
            "weighted_score"
        ] = same_scores


        cosine_scores = calculate_cosine_scores(

            candidate_df=
                same_model_df,

            year=
                year,

            predicted_price=
                predicted_price,

            km_driven=
                km_driven,

            mileage=
                mileage,

            engine=
                engine,

            seats=
                seats

        )


        same_model_df[
            "cosine_score"
        ] = (
            cosine_scores
            * 100
        )


        # Same-model weighted score already has
        # a theoretical maximum of 100.

        same_model_df[
            "final_score"
        ] = (

            same_model_df[
                "weighted_score"
            ]
            * 0.85

            +

            same_model_df[
                "cosine_score"
            ]
            * 0.15

        )


        same_model_df[
            "final_score"
        ] = (
            same_model_df[
                "final_score"
            ]
            .clip(
                lower=0,
                upper=100
            )
        )


        same_model_df = (
            same_model_df
            .sort_values(
                "final_score",
                ascending=False
            )
        )


    # ------------------------------------------------------
    # Build same-model results
    # ------------------------------------------------------

    similar_listings = []


    for _, car in (
        same_model_df
        .head(
            number_of_similar
        )
        .iterrows()
    ):

        similar_listings.append(

            create_result(

                car,

                user_year=
                    year,

                predicted_price=
                    predicted_price,

                user_km=
                    km_driven,

                user_fuel=
                    fuel,

                user_transmission=
                    transmission,

                user_engine=
                    engine,

                user_seats=
                    seats

            )

        )


    # ======================================================
    # PART B - ALTERNATIVE VEHICLES
    # ======================================================

    alternatives_df = df[

        df["model"] != model

    ].copy()


    # ======================================================
    # STRICT FILTER
    # ======================================================

    filtered_alternatives = alternatives_df[


        # Similar age

        (
            abs(
                alternatives_df["year"]
                -
                year
            )
            <= 4
        )


        &


        # Same fuel

        (
            alternatives_df["fuel"]
            ==
            fuel
        )


        &


        # Same transmission

        (
            alternatives_df["transmission"]
            ==
            transmission
        )


        &


        # Similar seating capacity

        (
            abs(
                alternatives_df["seats"]
                -
                seats
            )
            <= 2
        )


        &


        # Price from 60% of predicted value

        (
            alternatives_df[
                "selling_price"
            ]
            >=
            predicted_price * 0.60
        )


        &


        # Price up to 140% of predicted value

        (
            alternatives_df[
                "selling_price"
            ]
            <=
            predicted_price * 1.40
        )


    ].copy()


    # ======================================================
    # RELAX FILTER IF NECESSARY
    # ======================================================

    if (
        len(filtered_alternatives)
        <
        number_of_alternatives
    ):

        filtered_alternatives = alternatives_df[


            (
                abs(
                    alternatives_df["year"]
                    -
                    year
                )
                <= 5
            )


            &


            (
                alternatives_df["fuel"]
                ==
                fuel
            )


            &


            (
                alternatives_df["transmission"]
                ==
                transmission
            )


            &


            (
                alternatives_df[
                    "selling_price"
                ]
                >=
                predicted_price * 0.50
            )


            &


            (
                alternatives_df[
                    "selling_price"
                ]
                <=
                predicted_price * 1.50
            )


        ].copy()


    # ======================================================
    # ALTERNATIVE WEIGHTED SCORES
    # ======================================================

    alternative_scores = []


    for _, car in (
        filtered_alternatives
        .iterrows()
    ):

        score = calculate_match_score(

            car=car,

            brand=brand,
            model=model,
            year=year,
            predicted_price=predicted_price,
            km_driven=km_driven,
            fuel=fuel,
            seller_type=seller_type,
            transmission=transmission,
            owner=owner,
            mileage=mileage,
            engine=engine,
            seats=seats,

            same_model=False

        )

        alternative_scores.append(
            score
        )


    # ======================================================
    # ALTERNATIVE COSINE + WEIGHTED SCORE
    # ======================================================

    if not filtered_alternatives.empty:

        filtered_alternatives[
            "weighted_score"
        ] = alternative_scores


        cosine_scores = calculate_cosine_scores(

            candidate_df=
                filtered_alternatives,

            year=
                year,

            predicted_price=
                predicted_price,

            km_driven=
                km_driven,

            mileage=
                mileage,

            engine=
                engine,

            seats=
                seats

        )


        filtered_alternatives[
            "cosine_score"
        ] = (
            cosine_scores
            * 100
        )


        # Alternative score maximum:
        #
        # Brand          5
        # Fuel          10
        # Transmission  10
        # Owner          5
        # Seller         3
        # Year          10
        # Price         10
        # KM             7
        # Engine         3
        # Mileage        1
        # Seats          1
        #
        # TOTAL         65

        alternative_max_weighted_score = 65


        filtered_alternatives[
            "normalized_weighted_score"
        ] = (

            filtered_alternatives[
                "weighted_score"
            ]

            /

            alternative_max_weighted_score

            * 100

        )


        filtered_alternatives[
            "normalized_weighted_score"
        ] = (

            filtered_alternatives[
                "normalized_weighted_score"
            ]
            .clip(
                lower=0,
                upper=100
            )

        )


        # 85% rule-based weighted similarity
        # 15% cosine similarity

        filtered_alternatives[
            "final_score"
        ] = (

            filtered_alternatives[
                "normalized_weighted_score"
            ]
            * 0.85

            +

            filtered_alternatives[
                "cosine_score"
            ]
            * 0.15

        )


        filtered_alternatives[
            "final_score"
        ] = (

            filtered_alternatives[
                "final_score"
            ]
            .clip(
                lower=0,
                upper=100
            )

        )


        # Highest scoring recommendations first

        filtered_alternatives = (

            filtered_alternatives

            .sort_values(
                "final_score",
                ascending=False
            )

        )


        # Keep recommendations diverse.
        # Do not show several listings of the same
        # brand/model combination.

        filtered_alternatives = (

            filtered_alternatives

            .drop_duplicates(
                subset=[
                    "brand",
                    "model"
                ]
            )

        )


    # ======================================================
    # BUILD ALTERNATIVE RESULTS
    # ======================================================

    alternative_vehicles = []


    for _, car in (

        filtered_alternatives

        .head(
            number_of_alternatives
        )

        .iterrows()

    ):

        alternative_vehicles.append(

            create_result(

                car,

                user_year=
                    year,

                predicted_price=
                    predicted_price,

                user_km=
                    km_driven,

                user_fuel=
                    fuel,

                user_transmission=
                    transmission,

                user_engine=
                    engine,

                user_seats=
                    seats

            )

        )


    # ======================================================
    # RETURN
    # ======================================================

    return {

        "similar_listings":
            similar_listings,

        "alternative_vehicles":
            alternative_vehicles

    }