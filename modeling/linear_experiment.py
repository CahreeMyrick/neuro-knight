from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from preprocess import target_to_cp


class LinearExperiment:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.seed = cfg["split"]["random_state"]

    def make_model(self, base_model):
        if self.cfg["preprocessing"]["use_scaler"]:
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", base_model),
                ]
            )

        return base_model

    def get_models(self) -> dict:
        use_scaler = self.cfg["preprocessing"]["use_scaler"]

        ridge_param = "model__alpha" if use_scaler else "alpha"
        lasso_param = "model__alpha" if use_scaler else "alpha"

        return {
            "linear": {
                "model": self.make_model(LinearRegression()),
                "params": {},
            },
            "ridge": {
                "model": self.make_model(Ridge(random_state=self.seed)),
                "params": {
                    ridge_param: self.cfg["models"]["ridge"]["alphas"],
                },
            },
            "lasso": {
                "model": self.make_model(
                    Lasso(
                        max_iter=self.cfg["models"]["lasso"]["max_iter"],
                        tol=self.cfg["models"]["lasso"]["tol"],
                        random_state=self.seed,
                    )
                ),
                "params": {
                    lasso_param: self.cfg["models"]["lasso"]["alphas"],
                },
            },
        }

    def train_one(self, model, params: dict, X_train, y_train) -> dict:
        if not params:
            model.fit(X_train, y_train)

            return {
                "model": model,
                "best_params": {},
                "best_cv_mse": None,
            }

        cv = KFold(
            n_splits=self.cfg["training"]["cv_splits"],
            shuffle=True,
            random_state=self.seed,
        )

        search = GridSearchCV(
            estimator=model,
            param_grid=params,
            scoring="neg_mean_squared_error",
            cv=cv,
            n_jobs=self.cfg["training"]["n_jobs"],
            verbose=self.cfg["training"]["verbose"],
        )

        search.fit(X_train, y_train)

        return {
            "model": search.best_estimator_,
            "best_params": search.best_params_,
            "best_cv_mse": float(-search.best_score_),
        }

    def train(self, X_train, y_train) -> dict:
        trained = {}

        for name, config in self.get_models().items():
            print("\n" + "-" * 80)
            print(f"Training {name}")
            print("-" * 80)

            trained[name] = self.train_one(
                model=config["model"],
                params=config["params"],
                X_train=X_train,
                y_train=y_train,
            )

            print("Best params:", trained[name]["best_params"])
            print("Best CV MSE:", trained[name]["best_cv_mse"])

        return trained

    def evaluate(self, model, X_test, y_test) -> tuple[dict, np.ndarray]:
        pred = model.predict(X_test)

        mse_tanh = mean_squared_error(y_test, pred)
        mae_tanh = mean_absolute_error(y_test, pred)
        r2_tanh = r2_score(y_test, pred)

        cp_true = np.array(
            [target_to_cp(v, self.cfg) for v in y_test],
            dtype=np.float32,
        )

        cp_pred = np.array(
            [target_to_cp(v, self.cfg) for v in pred],
            dtype=np.float32,
        )

        mse_cp = mean_squared_error(cp_true, cp_pred)
        mae_cp = mean_absolute_error(cp_true, cp_pred)

        metrics = {
            "mse_tanh": float(mse_tanh),
            "mae_tanh": float(mae_tanh),
            "r2_tanh": float(r2_tanh),
            "mse_cp": float(mse_cp),
            "mae_cp": float(mae_cp),
        }

        return metrics, pred

    def export_model(self, model, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)

    def export_predictions(self, predictions: np.ndarray, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, predictions)

    def export_linear_weights(self, model, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(model, Pipeline):
            scaler = model.named_steps["scaler"]
            linear = model.named_steps["model"]

            coef = linear.coef_.reshape(-1)
            intercept = float(linear.intercept_)

            scale = scaler.scale_
            mean = scaler.mean_

            effective_coef = coef / scale
            effective_intercept = intercept - np.sum((coef * mean) / scale)

        else:
            linear = model
            effective_coef = linear.coef_.reshape(-1)
            effective_intercept = float(linear.intercept_)

        np.save(out_path.with_suffix(".coef.npy"), effective_coef)

        np.save(
            out_path.with_suffix(".intercept.npy"),
            np.array([effective_intercept], dtype=np.float32),
        )

        with open(out_path.with_suffix(".txt"), "w") as f:
            f.write("# effective linear model\n")
            f.write("# prediction = dot(coef, x) + intercept\n")
            f.write(f"intercept {effective_intercept:.10f}\n")

            for v in effective_coef:
                f.write(f"{float(v):.10f}\n")
