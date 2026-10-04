import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# ==========================================================
# 1. LOAD DATASET
# ==========================================================

DATA_PATH = "data/Car details v3.csv"

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)


# ==========================================================
# 2. REMOVE DUPLICATES
# ==========================================================

df = df.drop_duplicates().reset_index(drop=True)

print("Dataset after removing duplicates:", df.shape)


# ==========================================================
# 3. EXTRACT BRAND AND MODEL
# ==========================================================

# Example:
# Ford Fiesta 1.5 TDCi
# Brand = Ford
# Model = Fiesta

df["brand"] = df["name"].astype(str).str.split().str[0]

df["model"] = df["name"].astype(str).str.split().str[1]


# ==========================================================
# 4. CLEAN NUMERICAL VALUES
# ==========================================================

# Mileage example:
# "23.4 kmpl" -> 23.4

df["mileage_num"] = pd.to_numeric(
    df["mileage"]
    .astype(str)
    .str.extract(r"([\d.]+)")[0],
    errors="coerce"
)


# Engine example:
# "1498 CC" -> 1498

df["engine_num"] = pd.to_numeric(
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
# 5. SELECT FEATURES
# ==========================================================

features = [

    "brand",
    "model",
    "year",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "mileage_num",
    "engine_num",
    "seats"

]

target = "selling_price"


X = df[features]

y = df[target]


# ==========================================================
# 6. DEFINE CATEGORICAL AND NUMERICAL FEATURES
# ==========================================================

categorical_features = [

    "brand",
    "model",
    "fuel",
    "seller_type",
    "transmission",
    "owner"

]


numerical_features = [

    "year",
    "km_driven",
    "mileage_num",
    "engine_num",
    "seats"

]


# ==========================================================
# 7. PREPROCESSING PIPELINES
# ==========================================================

categorical_pipeline = Pipeline(

    steps=[

        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )

    ]

)


numerical_pipeline = Pipeline(

    steps=[

        (
            "imputer",
            SimpleImputer(strategy="median")
        )

    ]

)


preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),

        (
            "numerical",
            numerical_pipeline,
            numerical_features
        )

    ]

)


# ==========================================================
# 8. TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42

)


print("\nTraining records:", len(X_train))

print("Testing records:", len(X_test))


# ==========================================================
# 9. DEFINE MODELS
# ==========================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            random_state=42
        ),

    "Random Forest":
        RandomForestRegressor(

            n_estimators=100,

            random_state=42,

            n_jobs=-1

        )

}


# ==========================================================
# 10. TRAIN AND EVALUATE MODELS
# ==========================================================

results = []

trained_models = {}


for model_name, algorithm in models.items():

    print("\nTraining:", model_name)


    pipeline = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                algorithm
            )

        ]

    )


    pipeline.fit(
        X_train,
        y_train
    )


    predictions = pipeline.predict(
        X_test
    )


    mae = mean_absolute_error(
        y_test,
        predictions
    )


    rmse = np.sqrt(

        mean_squared_error(
            y_test,
            predictions
        )

    )


    r2 = r2_score(
        y_test,
        predictions
    )


    results.append({

        "Model": model_name,

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2

    })


    trained_models[
        model_name
    ] = pipeline


# ==========================================================
# 11. DISPLAY MODEL RESULTS
# ==========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(

    by="R2",

    ascending=False

)


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(results_df.to_string(index=False))


# Save results

results_df.to_csv(

    "model_results.csv",

    index=False

)


# ==========================================================
# 12. SELECT BEST MODEL
# ==========================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]


print(
    "\nBest Model:",
    best_model_name
)


# ==========================================================
# 13. CREATE MODELS DIRECTORY
# ==========================================================

os.makedirs(

    "models",

    exist_ok=True

)


# ==========================================================
# 14. SAVE BEST MODEL
# ==========================================================

joblib.dump(

    best_model,

    "models/price_model.pkl"

)


print(
    "Saved model to models/price_model.pkl"
)


# ==========================================================
# 15. CREATE BRAND -> MODEL MAPPING
# ==========================================================

brand_model_mapping = {}


for brand in sorted(

    df["brand"]
    .dropna()
    .unique()
):

    models_for_brand = (

        df.loc[
            df["brand"] == brand,
            "model"
        ]

        .dropna()

        .unique()

        .tolist()

    )


    brand_model_mapping[brand] = sorted(
        models_for_brand
    )


# ==========================================================
# 16. CREATE METADATA
# ==========================================================

metadata = {

    "brand":
        sorted(
            df["brand"]
            .dropna()
            .unique()
            .tolist()
        ),

    "brand_models":
        brand_model_mapping,

    "fuel":
        sorted(
            df["fuel"]
            .dropna()
            .unique()
            .tolist()
        ),

    "seller_type":
        sorted(
            df["seller_type"]
            .dropna()
            .unique()
            .tolist()
        ),

    "transmission":
        sorted(
            df["transmission"]
            .dropna()
            .unique()
            .tolist()
        ),

    "owner":
        sorted(
            df["owner"]
            .dropna()
            .unique()
            .tolist()
        ),

    "features":
        features,

    "best_model":
        best_model_name

}


# ==========================================================
# 17. SAVE METADATA
# ==========================================================

with open(

    "models/metadata.json",

    "w"

) as file:

    json.dump(

        metadata,

        file,

        indent=4

    )


print(
    "Saved metadata to models/metadata.json"
)


print("\nTraining completed successfully!")