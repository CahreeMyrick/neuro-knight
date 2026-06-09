import chess
import math
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

PATH1 = "/Users/cahree/dev/systems/chess/engines/engine_5/data/data_stockfish_labeled.csv"
PATH2 = "/Users/cahree/dev/systems/chess/engines/engine_5/data/data_stockfish_labeled_400k.csv"

OUTPUT_DIR = Path("/Users/cahree/dev/systems/chess/engines/engine_4/model_outputs/nn_sparse_769")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEN_COL = "fen"
TARGET_COL = "stockfish_cp"
GROUP_COL = "game_id"

RANDOM_STATE = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

BATCH_SIZE = 2048
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-5

EARLY_STOPPING_PATIENCE = 6
MIN_DELTA = 1e-5

INPUT_SIZE = 769
HIDDEN_SIZE = 32

CP_CLIP = 2000.0
TANH_SCALE = 400.0

# Optional:
# If True, rows with extreme mate-like scores are removed entirely before training.
DROP_EXTREME_ROWS = False
EXTREME_CP_THRESHOLD = 10000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    """
    768 board features + 1 side-to-move feature.
    Layout:
        index = square * 12 + color_offset + piece_type_index
    where piece_type_index:
        pawn=0, knight=1, bishop=2, rook=3, queen=4, king=5
    """
    board = chess.Board(fen)
    x = np.zeros(INPUT_SIZE, dtype=np.float32)

    for square, piece in board.piece_map().items():
        piece_type_index = PIECE_TO_INDEX[piece.piece_type]
        color_offset = 0 if piece.color == chess.WHITE else 6
        idx = square * 12 + color_offset + piece_type_index
        x[idx] = 1.0

    x[768] = 1.0 if board.turn == chess.WHITE else 0.0
    return x


class SparseEvalNet(nn.Module):
    def __init__(self, input_size=769, hidden_size=32):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        out = torch.tanh(self.fc2(h))
        return out


def evaluate_regression(model, loader):
    model.eval()

    ys_true = []
    ys_pred = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            pred = model(xb).squeeze(1)

            ys_true.append(yb.cpu().numpy())
            ys_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(ys_true)
    y_pred = np.concatenate(ys_pred)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    cp_true = np.array([target_to_cp(v) for v in y_true], dtype=np.float32)
    cp_pred = np.array([target_to_cp(v) for v in y_pred], dtype=np.float32)

    cp_mse = mean_squared_error(cp_true, cp_pred)
    cp_mae = mean_absolute_error(cp_true, cp_pred)

    return {
        "mse_tanh": float(mse),
        "mae_tanh": float(mae),
        "r2_tanh": float(r2),
        "mse_cp": float(cp_mse),
        "mae_cp": float(cp_mae),
    }


def export_weights_to_txt(model: SparseEvalNet, out_path: Path):
    """
    Export flat order:
    W1 (32 x 769), b1 (32), W2 (1 x 32), b2 (1)
    """
    sd = model.state_dict()

    W1 = sd["fc1.weight"].cpu().numpy()
    b1 = sd["fc1.bias"].cpu().numpy()
    W2 = sd["fc2.weight"].cpu().numpy()
    b2 = sd["fc2.bias"].cpu().numpy()

    with open(out_path, "w") as f:
        for v in W1.flatten():
            f.write(f"{float(v):.10f}\n")
        for v in b1.flatten():
            f.write(f"{float(v):.10f}\n")
        for v in W2.flatten():
            f.write(f"{float(v):.10f}\n")
        for v in b2.flatten():
            f.write(f"{float(v):.10f}\n")


def export_weights_to_header(model: SparseEvalNet, out_path: Path):
    sd = model.state_dict()

    W1 = sd["fc1.weight"].cpu().numpy()
    b1 = sd["fc1.bias"].cpu().numpy()
    W2 = sd["fc2.weight"].cpu().numpy()
    b2 = sd["fc2.bias"].cpu().numpy()

    def arr1(name, arr):
        vals = ", ".join(f"{float(v):.8f}f" for v in arr)
        return f"inline constexpr float {name}[{len(arr)}] = {{{vals}}};\n"

    def arr2(name, arr):
        rows = []
        for row in arr:
            vals = ", ".join(f"{float(v):.8f}f" for v in row)
            rows.append(f"    {{{vals}}}")
        joined = ",\n".join(rows)
        return (
            f"inline constexpr float {name}[{arr.shape[0]}][{arr.shape[1]}] = {{\n"
            f"{joined}\n"
            f"}};\n"
        )

    content = []
    content.append("#pragma once\n\n")
    content.append("namespace nn_weights {\n\n")
    content.append(f"inline constexpr int INPUT_SIZE = {INPUT_SIZE};\n")
    content.append(f"inline constexpr int HIDDEN_SIZE = {HIDDEN_SIZE};\n\n")
    content.append(arr2("W1", W1))
    content.append("\n")
    content.append(arr1("b1", b1))
    content.append("\n")
    content.append(arr2("W2", W2))
    content.append("\n")
    content.append(arr1("b2", b2))
    content.append("\n")
    content.append("} // namespace nn_weights\n")

    with open(out_path, "w") as f:
        f.write("".join(content))


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
    """
    Split into train / val / test by game_id to avoid leakage.
    """

    if not math.isclose(TRAIN_RATIO + VAL_RATIO + TEST_RATIO, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0")

    # First: train vs temp
    temp_ratio = VAL_RATIO + TEST_RATIO

    splitter1 = GroupShuffleSplit(
        n_splits=1,
        test_size=temp_ratio,
        random_state=RANDOM_STATE,
    )

    train_idx, temp_idx = next(splitter1.split(df, groups=df[GROUP_COL]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    # Second: val vs test within temp
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
# SPLIT BY GAME
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
# ENCODE DATA
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
# DATALOADERS
# ============================================================

train_ds = TensorDataset(
    torch.from_numpy(X_train),
    torch.from_numpy(y_train),
)

val_ds = TensorDataset(
    torch.from_numpy(X_val),
    torch.from_numpy(y_val),
)

test_ds = TensorDataset(
    torch.from_numpy(X_test),
    torch.from_numpy(y_test),
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


# ============================================================
# TRAIN
# ============================================================

print("=" * 80)
print("TRAINING")
print("=" * 80)

model = SparseEvalNet(INPUT_SIZE, HIDDEN_SIZE).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.MSELoss()

best_state = None
best_metric = float("inf")
history = []
patience_counter = 0

print("Training on:", DEVICE)

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0

    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()
        pred = model(xb).squeeze(1)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)

    train_loss = running_loss / len(train_ds)
    val_metrics = evaluate_regression(model, val_loader)

    row = {
        "epoch": epoch,
        "train_loss": float(train_loss),
        **{f"val_{k}": v for k, v in val_metrics.items()},
    }
    history.append(row)

    print(
        f"Epoch {epoch:02d} | "
        f"train_loss={train_loss:.6f} | "
        f"val_mse_tanh={val_metrics['mse_tanh']:.6f} | "
        f"val_mae_cp={val_metrics['mae_cp']:.3f}"
    )

    current_metric = val_metrics["mse_tanh"]

    if current_metric < best_metric - MIN_DELTA:
        best_metric = current_metric
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
        print(f"  -> new best model saved (val_mse_tanh={best_metric:.6f})")
    else:
        patience_counter += 1
        print(f"  -> no improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE})")

    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print("Early stopping triggered.")
        break

if best_state is None:
    raise RuntimeError("Training failed: no best state recorded.")

model.load_state_dict(best_state)


# ============================================================
# FINAL EVALUATION
# ============================================================

print("=" * 80)
print("FINAL EVALUATION")
print("=" * 80)

val_metrics = evaluate_regression(model, val_loader)
test_metrics = evaluate_regression(model, test_loader)

print("Best validation metrics:")
print(val_metrics)

print("Final test metrics:")
print(test_metrics)


# ============================================================
# SAVE ARTIFACTS
# ============================================================

torch.save(model.state_dict(), OUTPUT_DIR / "best_sparse_eval_769.pt")
joblib.dump(history, OUTPUT_DIR / "history.joblib")

history_df = pd.DataFrame(history)
history_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)

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
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
        "MIN_DELTA": MIN_DELTA,
        "INPUT_SIZE": INPUT_SIZE,
        "HIDDEN_SIZE": HIDDEN_SIZE,
        "CP_CLIP": CP_CLIP,
        "TANH_SCALE": TANH_SCALE,
        "DROP_EXTREME_ROWS": DROP_EXTREME_ROWS,
        "EXTREME_CP_THRESHOLD": EXTREME_CP_THRESHOLD,
        "DEVICE": DEVICE,
    },
    "validation_metrics": val_metrics,
    "test_metrics": test_metrics,
    "dataset_sizes": {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_games": int(train_df[GROUP_COL].nunique()),
        "val_games": int(val_df[GROUP_COL].nunique()),
        "test_games": int(test_df[GROUP_COL].nunique()),
    },
}

with open(OUTPUT_DIR / "final_metrics.json", "w") as f:
    json.dump(metrics_out, f, indent=2)

export_weights_to_txt(model, OUTPUT_DIR / "weights_sparse_769.txt")
export_weights_to_header(model, OUTPUT_DIR / "nn_weights.hpp")

print("=" * 80)
print("DONE")
print("=" * 80)
print("Saved artifacts to:", OUTPUT_DIR)