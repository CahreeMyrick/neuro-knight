import json
import math
from pathlib import Path

import chess
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIG
# ============================================================

PATH1 = "/Users/cahree/dev/systems/chess/engines/engine_5/data/data_stockfish_labeled.csv"
PATH2 = "/Users/cahree/dev/systems/chess/engines/engine_5/data/data_stockfish_labeled_400k.csv"

OUTPUT_DIR = Path("/Users/cahree/dev/systems/chess/engines/neuro-knight/model_outputs/scratch/linear_models_769")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEN_COL = "fen"
TARGET_COL = "stockfish_cp"
GROUP_COL = "game_id"

RANDOM_STATE = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

INPUT_SIZE = 769

CP_CLIP = 2000.0
TANH_SCALE = 400.0

DROP_EXTREME_ROWS = False
EXTREME_CP_THRESHOLD = 10000

USE_SCALER = True


# ============================================================
# HELPERS
# ============================================================

PIECE_TO_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def cp_to_target(cp: float) -> float:
    cp = max(-CP_CLIP, min(CP_CLIP, float(cp)))
    return math.tanh(cp / TANH_SCALE)


def target_to_cp(y: float) -> float:
    y = max(-0.999, min(0.999, float(y)))
    return float(TANH_SCALE * np.arctanh(y))


def encode_fen_769(fen: str) -> np.ndarray:
    board = chess.Board(fen)
    x = np.zeros(INPUT_SIZE, dtype=np.float32)

    for square, piece in board.piece_map().items():
        piece_type_index = PIECE_TO_INDEX[piece.piece_type]
        color_offset = 0 if piece.color == chess.WHITE else 6
        idx = square * 12 + color_offset + piece_type_index
        x[idx] = 1.0

    x[768] = 1.0 if board.turn == chess.WHITE else 0.0
    return x


def build_dataset(frame: pd.DataFrame):
    X = np.zeros((len(frame), INPUT_SIZE), dtype=np.float32)
    y = np.zeros(len(frame), dtype=np.float32)

    for i, row in enumerate(frame.itertuples(index=False)):
        fen = getattr(row, FEN_COL)
        cp = getattr(row, TARGET_COL)

        X[i] = encode_fen_769(fen)
        y[i] = cp_to_target(cp)

        if i > 0 and i % 50000 == 0:
            print(f"Encoded {i}/{len(frame)} rows")

    return X, y


def split_by_game(df: pd.DataFrame):
    if not math.isclose(TRAIN_RATIO + VAL_RATIO + TEST_RATIO, 1.0):
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0")

    temp_ratio = VAL_RATIO + TEST_RATIO

    splitter1 = GroupShuffleSplit(
        n_splits=1,
        test_size=temp_ratio,
        random_state=RANDOM_STATE,
    )

    train_idx, temp_idx = next(splitter1.split(df, groups=df[GROUP_COL]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    test_fraction_within_temp = TEST_RATIO / temp_ratio

    splitter2 = GroupShuffleSplit(
        n_splits=1,
        test_size=test_fraction_within_temp,
        random_state=RANDOM_STATE,
    )

    val_idx, test_idx = next(splitter2.split(temp_df, groups=temp_df[GROUP_COL]))

    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df


def evaluate_model(model, X, y):
    pred = model.predict(X)

    mse_tanh = mean_squared_error(y, pred)
    mae_tanh = mean_absolute_error(y, pred)
    r2_tanh = r2_score(y, pred)

    cp_true = np.array([target_to_cp(v) for v in y], dtype=np.float32)
    cp_pred = np.array([target_to_cp(v) for v in pred], dtype=np.float32)

    mse_cp = mean_squared_error(cp_true, cp_pred)
    mae_cp = mean_absolute_error(cp_true, cp_pred)

    return {
        "mse_tanh": float(mse_tanh),
        "mae_tanh": float(mae_tanh),
        "r2_tanh": float(r2_tanh),
        "mse_cp": float(mse_cp),
        "mae_cp": float(mae_cp),
    }, pred


def make_model(base_model):
    if USE_SCALER:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", base_model),
        ])

    return base_model


def export_linear_weights(model, out_path: Path):
    """
    Exports effective weights for C++ if model is linear/ridge/lasso.

    If scaler is used:
        y = coef dot ((x - mean) / scale) + intercept
    is converted into:
        y = effective_coef dot x + effective_intercept
    """
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
    np.save(out_path.with_suffix(".intercept.npy"), np.array([effective_intercept], dtype=np.float32))

    with open(out_path.with_suffix(".txt"), "w") as f:
        f.write("# effective linear model\n")
        f.write("# prediction = dot(coef, x) + intercept\n")
        f.write(f"intercept {effective_intercept:.10f}\n")
        for v in effective_coef:
            f.write(f"{float(v):.10f}\n")


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("LOADING DATA")
print("=" * 80)

df1 = pd.read_csv(PATH1)
df2 = pd.read_csv(PATH2)
df = pd.concat([df1, df2], ignore_index=True)

print("Original combined shape:", df.shape)

required_cols = [FEN_COL, TARGET_COL, GROUP_COL]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df[[FEN_COL, TARGET_COL, GROUP_COL]].dropna().reset_index(drop=True)
print("After selecting required columns:", df.shape)

df = df.drop_duplicates().reset_index(drop=True)
print("After dropping exact duplicates:", df.shape)

if DROP_EXTREME_ROWS:
    before = len(df)
    df = df[df[TARGET_COL].abs() < EXTREME_CP_THRESHOLD].reset_index(drop=True)
    after = len(df)
    print(f"After dropping extreme rows |cp| >= {EXTREME_CP_THRESHOLD}: {df.shape}")
    print(f"Rows removed: {before - after}")

print(f"Unique games: {df[GROUP_COL].nunique()}")


# ============================================================
# SPLIT
# ============================================================

print("=" * 80)
print("SPLITTING BY GAME")
print("=" * 80)

train_df, val_df, test_df = split_by_game(df)

print("Train shape:", train_df.shape)
print("Val shape  :", val_df.shape)
print("Test shape :", test_df.shape)

print("Unique train games:", train_df[GROUP_COL].nunique())
print("Unique val games  :", val_df[GROUP_COL].nunique())
print("Unique test games :", test_df[GROUP_COL].nunique())


# ============================================================
# ENCODE
# ============================================================

print("=" * 80)
print("ENCODING TRAIN")
print("=" * 80)
X_train, y_train = build_dataset(train_df)

print("=" * 80)
print("ENCODING VAL")
print("=" * 80)
X_val, y_val = build_dataset(val_df)

print("=" * 80)
print("ENCODING TEST")
print("=" * 80)
X_test, y_test = build_dataset(test_df)

np.save(OUTPUT_DIR / "X_train.npy", X_train)
np.save(OUTPUT_DIR / "y_train.npy", y_train)
np.save(OUTPUT_DIR / "X_val.npy", X_val)
np.save(OUTPUT_DIR / "y_val.npy", y_val)
np.save(OUTPUT_DIR / "X_test.npy", X_test)
np.save(OUTPUT_DIR / "y_test.npy", y_test)


# ============================================================
# TRAIN LINEAR MODELS
# ============================================================

print("=" * 80)
print("TRAINING LINEAR MODELS")
print("=" * 80)

models = {
    "linear": {
        "model": make_model(LinearRegression()),
        "params": {},
    },
    "ridge": {
        "model": make_model(Ridge(random_state=RANDOM_STATE)),
        "params": {
            "model__alpha" if USE_SCALER else "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
    },
    "lasso": {
        "model": make_model(Lasso(max_iter=10000, tol=1e-4, random_state=RANDOM_STATE)),
        "params": {
            "model__alpha" if USE_SCALER else "alpha": [0.0001, 0.001, 0.01, 0.1],
        },
    },
}

cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

results = {}

for name, config in models.items():
    print("\n" + "-" * 80)
    print(f"Training {name}")
    print("-" * 80)

    model = config["model"]
    params = config["params"]

    if params:
        search = GridSearchCV(
            estimator=model,
            param_grid=params,
            scoring="neg_mean_squared_error",
            cv=cv,
            n_jobs=1,
            verbose=1,
        )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        print("Best params:", search.best_params_)
        print("Best CV MSE:", -search.best_score_)
        best_params = search.best_params_
        best_cv_mse = float(-search.best_score_)
    else:
        best_model = model
        best_model.fit(X_train, y_train)

        best_params = {}
        best_cv_mse = None

    val_metrics, val_pred = evaluate_model(best_model, X_val, y_val)
    test_metrics, test_pred = evaluate_model(best_model, X_test, y_test)

    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)

    results[name] = {
        "best_params": best_params,
        "best_cv_mse": best_cv_mse,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    joblib.dump(best_model, OUTPUT_DIR / f"{name}_model.joblib")
    np.save(OUTPUT_DIR / f"{name}_val_pred.npy", val_pred)
    np.save(OUTPUT_DIR / f"{name}_test_pred.npy", test_pred)

    if name in ["linear", "ridge", "lasso"]:
        export_linear_weights(best_model, OUTPUT_DIR / f"{name}_effective_weights")


# ============================================================
# SAVE RESULTS
# ============================================================

metrics_out = {
    "config": {
        "PATH1": PATH1,
        "PATH2": PATH2,
        "FEN_COL": FEN_COL,
        "TARGET_COL": TARGET_COL,
        "GROUP_COL": GROUP_COL,
        "TRAIN_RATIO": TRAIN_RATIO,
        "VAL_RATIO": VAL_RATIO,
        "TEST_RATIO": TEST_RATIO,
        "INPUT_SIZE": INPUT_SIZE,
        "CP_CLIP": CP_CLIP,
        "TANH_SCALE": TANH_SCALE,
        "DROP_EXTREME_ROWS": DROP_EXTREME_ROWS,
        "EXTREME_CP_THRESHOLD": EXTREME_CP_THRESHOLD,
        "USE_SCALER": USE_SCALER,
    },
    "dataset_sizes": {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_games": int(train_df[GROUP_COL].nunique()),
        "val_games": int(val_df[GROUP_COL].nunique()),
        "test_games": int(test_df[GROUP_COL].nunique()),
    },
    "results": results,
}

with open(OUTPUT_DIR / "linear_model_results.json", "w") as f:
    json.dump(metrics_out, f, indent=2)

joblib.dump(results, OUTPUT_DIR / "linear_model_results.joblib")

print("=" * 80)
print("DONE")
print("=" * 80)

for name, result in results.items():
    print(f"\n{name}")
    print("Best params:", result["best_params"])
    print("Validation:", result["validation_metrics"])
    print("Test:", result["test_metrics"])

print("\nSaved outputs to:", OUTPUT_DIR)