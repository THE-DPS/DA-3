from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "day_cleaned.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

TARGET = "cnt"
CATEGORICAL = ["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]
NUMERIC = ["temp", "atemp", "hum", "windspeed"]
LEAKAGE_OR_ID = {"instant", "dteday", "casual", "registered", TARGET}

def metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }

def main():
    if not DATA.exists():
        raise FileNotFoundError("Run `python src/prepare_data.py` first.")
    df = pd.read_csv(DATA, parse_dates=["dteday"]).sort_values("dteday").reset_index(drop=True)
    features = [c for c in CATEGORICAL + NUMERIC if c in df.columns]
    if not features:
        raise ValueError("No expected predictor columns found.")
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split].copy(), df.iloc[split:].copy()
    X_train, y_train = train[features], train[TARGET]
    X_test, y_test = test[features], test[TARGET]

    cat_cols = [c for c in CATEGORICAL if c in features]
    num_cols = [c for c in NUMERIC if c in features]
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    prep = ColumnTransformer([("cat", cat_pipe, cat_cols), ("num", num_pipe, num_cols)])

    results = []
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    results.append({"model":"Training mean baseline", "cv_MAE": None, **metrics(y_test, baseline.predict(X_test))})

    linear = Pipeline([("prep", prep), ("model", LinearRegression())])
    tscv = TimeSeriesSplit(n_splits=5)
    linear_cv = -np.mean(__import__("sklearn.model_selection").model_selection.cross_val_score(
        linear, X_train, y_train, cv=tscv, scoring="neg_mean_absolute_error"))
    linear.fit(X_train, y_train)
    results.append({"model":"Linear Regression", "cv_MAE":float(linear_cv), **metrics(y_test, linear.predict(X_test))})

    rf = Pipeline([("prep", prep), ("model", RandomForestRegressor(random_state=42, n_jobs=-1))])
    grid = {"model__n_estimators":[200, 400], "model__max_depth":[None, 10, 20],
            "model__min_samples_leaf":[1, 3]}
    search = GridSearchCV(rf, grid, cv=TimeSeriesSplit(n_splits=5),
                          scoring="neg_mean_absolute_error", n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    pred = best.predict(X_test)
    results.append({"model":"Random Forest (tuned)", "cv_MAE":float(-search.best_score_),
                    **metrics(y_test, pred), "best_params":search.best_params_})

    pd.DataFrame(results).to_csv(OUT / "model_metrics.csv", index=False)
    predictions = pd.DataFrame({"date":test["dteday"].dt.strftime("%Y-%m-%d"),
                                "actual":y_test.to_numpy(), "predicted":pred,
                                "residual":y_test.to_numpy()-pred})
    predictions.to_csv(OUT / "test_predictions.csv", index=False)
    payload = {
        "source":"UCI Bike Sharing Dataset (daily file)",
        "test_start":str(test["dteday"].min().date()),
        "test_end":str(test["dteday"].max().date()),
        "train_rows":int(len(train)), "test_rows":int(len(test)),
        "target":TARGET, "features":features,
        "excluded_features":sorted(LEAKAGE_OR_ID),
        "random_forest_best_params":search.best_params_,
        "metrics":results,
        "note":"Historical single-city data; not a current operational forecast."
    }
    (OUT / "model_metrics.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    joblib.dump(best, OUT / "best_model.joblib")
    print(json.dumps(payload, indent=2, default=str))
    print(f"Metrics saved to {OUT / 'model_metrics.json'}")

if __name__ == "__main__":
    main()
