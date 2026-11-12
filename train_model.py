import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train():
    processed_path = "data/processed/water_demand_clean.csv"

    if not os.path.exists(processed_path):
        from prepare_dataset import prepare_data
        prepare_data(output_path=processed_path)

    df = pd.read_csv(processed_path)

    if df.empty or df["daily_litres"].isna().any():
        raise ValueError("Dataset is empty or contains missing target values.")

    X = df.drop(columns=["daily_litres"])
    y = df["daily_litres"]

    # Keep the final test set separate from model selection.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    numeric_features = ["household_size", "rooms"]
    categorical_features = [
        "dwelling_type",
        "washing_machine",
        "garden",
        "water_source"
    ]

    def create_pipeline(estimator):
        # Preprocessing is fitted separately within each CV fold.
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                (
                    "cat",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    ),
                    categorical_features
                )
            ]
        )

        return Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", estimator)
        ])

    models = {
        "Mean Baseline": DummyRegressor(strategy="mean"),

        "Linear Regression": LinearRegression(),

        "Ridge Regression": Ridge(alpha=10.0),

        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_split=4,
            min_samples_leaf=3,
            random_state=42
        ),

        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.05,
            min_samples_leaf=4,
            loss="huber",
            random_state=42
        )
    }

    # Cross-validation uses training data only.
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    pipelines = {}

    for name, estimator in models.items():
        pipe = create_pipeline(estimator)

        cv_scores = cross_val_score(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1
        )

        mean_cv_mae = -cv_scores.mean()
        std_cv_mae = cv_scores.std()

        results.append({
            "Model": name,
            "CV_MAE": round(float(mean_cv_mae), 2),
            "CV_MAE_Std": round(float(std_cv_mae), 2)
        })

        pipelines[name] = pipe

    results_df = pd.DataFrame(results).sort_values("CV_MAE")

    os.makedirs("reports", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("\nCross-validation results (lower MAE is better):")
    print(results_df.to_string(index=False))

    # Select using CV only, not test-set performance.
    best_model_name = results_df.iloc[0]["Model"]
    best_pipeline = pipelines[best_model_name]

    # Fit the selected model on the full training set.
    best_pipeline.fit(X_train, y_train)

    # Evaluate once on the untouched test set.
    y_pred = best_pipeline.predict(X_test)

    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)

    print(f"\nSelected model: {best_model_name}")
    print(f"Test MAE:  {test_mae:.2f} litres/day")
    print(f"Test RMSE: {test_rmse:.2f} litres/day")
    print(f"Test R2:   {test_r2:.4f}")

    # Save the CV comparison report.
    results_df.to_csv(
        "reports/model_comparison.csv", index=False
    )

    # Save final test metrics separately.
    test_results = {
        "model_name": best_model_name,
        "test_MAE": round(float(test_mae), 2),
        "test_RMSE": round(float(test_rmse), 2),
        "test_R2": round(float(test_r2), 4),
        "test_records": int(len(y_test))
    }

    with open("reports/final_test_metrics.json", "w") as f:
        json.dump(test_results, f, indent=4)

    # Keep the same model path for Flask compatibility.
    joblib.dump(
        best_pipeline,
        "models/water_demand_pipeline.joblib"
    )

    metadata = {
        **test_results,
        "features": list(X.columns),
        "cv_folds": 5,
        "target": "daily_litres"
    }

    with open("models/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("\n[✓] Best model saved successfully.")
    print("[✓] Comparison: reports/model_comparison.csv")
    print("[✓] Test metrics: reports/final_test_metrics.json")


if __name__ == "__main__":
    train()