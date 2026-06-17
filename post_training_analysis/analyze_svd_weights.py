from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "model_outputs" / "experiments"

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M")
EXP_DIR = BASE / TIMESTAMP
# EXP_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = BASE / Path("2026-06-15_1600_EvalNet/model_weights.pt")


PIECE_NAMES = [
    "White Pawn", "White Knight", "White Bishop", "White Rook", "White Queen", "White King",
    "Black Pawn", "Black Knight", "Black Bishop", "Black Rook", "Black Queen", "Black King",
]

def vector_to_square_importance(v_768):

    square_feature = v_768.reshape(64, 12)

    square_importance = np.abs(square_feature).sum(axis=1)

    board = np.zeros((8,8), dtype=np.float32)

    for square in range(64):
        rank = square // 8
        file = square % 8

        row = 7-rank
        col = file

        board[row, col] = square_importance[square]

    return board

def vector_to_piece_importance(v_768):
    """
    Sum absolute importance by piece/color channel.
    """
    square_feature = v_768.reshape(64, 12)
    return np.abs(square_feature).sum(axis=0)

def split_board_and_stm(v_769):
    """
    Input vector layout:
        0..767 = board features
        768    = side-to-move feature
    """
    board_v = v_769[:768]
    stm_v = v_769[768]
    return board_v, stm_v

def load_w1(path):
    state = torch.load(path)
    return state["fc1.weight"].detach().cpu().numpy()

def plot_board_heatmap(board, title, filename):
    plt.figure(figsize=(7, 6))
    plt.imshow(board)
    plt.colorbar(label="Importance")
    plt.title(title)

    files = ["a", "b", "c", "d", "e", "f", "g", "h"]
    ranks = ["8", "7", "6", "5", "4", "3", "2", "1"]

    plt.xticks(range(8), files)
    plt.yticks(range(8), ranks)

    plt.xlabel("File")
    plt.ylabel("Rank")

    plt.show()

def plot_piece_importance(piece_importance, title, filename):
    plt.figure(figsize=(9, 5))
    plt.bar(range(12), piece_importance)
    plt.xticks(range(12), PIECE_NAMES, rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.title(title)

    plt.show()
    
def main():

    W1 = load_w1(MODEL_PATH)
    print("W1 shape:", W1.shape)

    print("\nComputing SVD...")
    U, S, Vt = np.linalg.svd(W1, full_matrices=False)

    print(Vt.shape)

    top_k = 8
    for k in range(top_k):
        v_769 = Vt[k]
        v_board, v_stm = split_board_and_stm(v_769)

        board_importance = vector_to_square_importance(v_board)
        piece_importance = vector_to_piece_importance(v_board)

        print(board_importance)

        plot_board_heatmap(
            board_importance,
            f"Square Importance: Singular Direction {k+1}",
            f"svd_direction_{k+1}_board.png",
        )

        plot_piece_importance(
            piece_importance,
            f"Piece-Type Importance: Singular Direction {k+1}",
            f"svd_direction_{k+1}_pieces.png",
        )


if __name__ == "__main__":
    main()


