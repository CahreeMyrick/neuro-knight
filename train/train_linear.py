from __future__ import annotations

from pathlib import Path
import chess
import numpy as np
import pandas as pd


class ChessDataset:
    def __init__(self, cfg: dict):
        self.cfg = cfg

        self.paths = cfg["dataset"]["paths"]
        self.fen_col = cfg["dataset"]["fen_col"]
        self.target_col = cfg["dataset"]["target_col"]
        self.group_col = cfg["dataset"]["group_col"]

        self.input_size = cfg["preprocessing"]["input_size"]
        self.cp_clip = cfg["preprocessing"]["cp_clip"]
        self.tanh_scale = cfg["preprocessing"]["tanh_scale"]
        self.drop_extreme_rows = cfg["preprocessing"]["drop_extreme_rows"]
        self.extreme_cp_threshold = cfg["preprocessing"]["extreme_cp_threshold"]

        self.train_ratio = cfg["split"]["train_ratio"]
        self.val_ratio = cfg["split"]["val_ratio"]
        self.test_ratio = cfg["split"]["test_ratio"]
        self.random_state = cfg["split"]["random_state"]

class LinearModelExperiment:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.seed = cfg["split"]["random_state"]

        exp_name = cfg["experiment"]["name"]
        self.exp_id = f"{exp_name}_{uuid.4().hex[:8]}"
        self.timestamp = datetime.now().isoformat(timespec="seconds")

        self.out
