import math
import numpy as np
import pandas as pd
import chess

from sklearn.model_selection import GroupShuffleSplit


PIECE_TO_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

def load_data(cfg: dict) -> pd.DataFrame:
    paths = cfg["dataset"]["paths"]

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)

    return df

def cp_to_target(cp: float, cfg: dict) -> float:
    cp_clip = cfg["preprocessing"]["cp_clip"]
    tanh_scale = cfg["preprocessing"]["tanh_scale"]

    cp = max(-cp_clip, min(cp_clip, float(cp)))
    return math.tanh(cp / tanh_scale)


def target_to_cp(y: float, cfg: dict) -> float:
    tanh_scale = cfg["preprocessing"]["tanh_scale"]

    y = max(-0.999, min(0.999, float(y)))
    return float(tanh_scale * np.arctanh(y))


def encode_fen_769(fen: str, cfg: dict) -> np.ndarray:
    input_size = cfg["preprocessing"]["input_size"]

    board = chess.Board(fen)
    x = np.zeros(input_size, dtype=np.float32)

    for square, piece in board.piece_map().items():
        piece_type_index = PIECE_TO_INDEX[piece.piece_type]
        color_offset = 0 if piece.color == chess.WHITE else 6
        idx = square * 12 + color_offset + piece_type_index
        x[idx] = 1.0

    x[768] = 1.0 if board.turn == chess.WHITE else 0.0
    return x


def build_dataset(frame: pd.DataFrame, cfg: dict):
    input_size = cfg["preprocessing"]["input_size"]
    fen_col = cfg["dataset"]["fen_col"]
    target_col = cfg["dataset"]["target_col"]

    X = np.zeros((len(frame), input_size), dtype=np.float32)
    y = np.zeros(len(frame), dtype=np.float32)

    for i, row in enumerate(frame.itertuples(index=False)):
        fen = getattr(row, fen_col)
        cp = getattr(row, target_col)

        X[i] = encode_fen_769(fen, cfg)
        y[i] = cp_to_target(cp, cfg)

        if i > 0 and i % 50000 == 0:
            print(f"Encoded {i}/{len(frame)} rows")

    return X, y


def split_by_game(df: pd.DataFrame, cfg: dict):
    train_ratio = cfg["split"]["train_ratio"]
    val_ratio = cfg["split"]["val_ratio"]
    test_ratio = cfg["split"]["test_ratio"]
    random_state = cfg["split"]["random_state"]
    group_col = cfg["dataset"]["group_col"]

    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    temp_ratio = val_ratio + test_ratio

    splitter1 = GroupShuffleSplit(
        n_splits=1,
        test_size=temp_ratio,
        random_state=random_state,
    )

    train_idx, temp_idx = next(splitter1.split(df, groups=df[group_col]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    test_fraction_within_temp = test_ratio / temp_ratio

    splitter2 = GroupShuffleSplit(
        n_splits=1,
        test_size=test_fraction_within_temp,
        random_state=random_state,
    )

    val_idx, test_idx = next(splitter2.split(temp_df, groups=temp_df[group_col]))

    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df
