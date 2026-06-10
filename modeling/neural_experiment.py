from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import TensorDataset, DataLoader

from preprocess import target_to_cp

from neural_model import EvalNet

class NeuralExperiment:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.seed = cfg["split"]["random_state"]
        self.device = self.resolve_device(cfg["training"]["device"])

        self.set_seed(self.seed)

    def resolve_device(self, requested_device: str) -> str:
        if requested_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return requested_device

    def set_seed(self, seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def make_model(self) -> EvalNet:
        return EvalNet(
            input_size=self.cfg["model"]["input_size"],
            hidden_size=self.cfg["model"]["hidden_size"],
        ).to(self.device)

    def make_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool,
    ) -> tuple[TensorDataset, DataLoader]:
        dataset = TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y),
        )

        loader = DataLoader(
            dataset,
            batch_size=self.cfg["training"]["batch_size"],
            shuffle=shuffle,
        )

        return dataset, loader

    def make_train_val_loaders(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
    ):
        train_ds, train_loader = self.make_dataloader(
            X_train,
            y_train,
            shuffle=True,
        )

        val_ds, val_loader = self.make_dataloader(
            X_val,
            y_val,
            shuffle=False,
        )

        return train_ds, val_ds, train_loader, val_loader

    def train(
        self,
        model: nn.Module,
        train_ds: TensorDataset,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> tuple[nn.Module, list[dict]]:
        print("=" * 80)
        print("TRAINING NEURAL NETWORK")
        print("=" * 80)
        print("Training on:", self.device)

        lr = self.cfg["training"]["lr"]
        weight_decay = self.cfg["training"]["weight_decay"]
        epochs = self.cfg["training"]["epochs"]
        patience = self.cfg["training"]["early_stopping_patience"]
        min_delta = self.cfg["training"]["min_delta"]

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        criterion = nn.MSELoss()

        history = []
        best_state = None
        best_metric = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()

                pred = model(xb).squeeze(1)
                loss = criterion(pred, yb)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * xb.size(0)

            train_loss = running_loss / len(train_ds)
            val_metrics, _ = self.evaluate(model, val_loader)

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

            if current_metric < best_metric - min_delta:
                best_metric = current_metric
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                patience_counter = 0
                print(f"  -> new best model saved")

            else:
                patience_counter += 1
                print(f"  -> no improvement ({patience_counter}/{patience})")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        if best_state is None:
            raise RuntimeError("Training failed: no best model was recorded.")

        model.load_state_dict(best_state)

        return model, history

    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
    ) -> tuple[dict, np.ndarray]:
        model.eval()

        ys_true = []
        ys_pred = []

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                pred = model(xb).squeeze(1)

                ys_true.append(yb.cpu().numpy())
                ys_pred.append(pred.cpu().numpy())

        y_true = np.concatenate(ys_true)
        y_pred = np.concatenate(ys_pred)

        mse_tanh = mean_squared_error(y_true, y_pred)
        mae_tanh = mean_absolute_error(y_true, y_pred)
        r2_tanh = r2_score(y_true, y_pred)

        cp_true = np.array(
            [target_to_cp(v, self.cfg) for v in y_true],
            dtype=np.float32,
        )

        cp_pred = np.array(
            [target_to_cp(v, self.cfg) for v in y_pred],
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

        return metrics, y_pred

    def export_model(self, model: nn.Module, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)

    def export_history(self, history: list[dict], outdir: Path) -> None:
        outdir.mkdir(parents=True, exist_ok=True)

        joblib.dump(history, outdir / "history.joblib")

        history_df = pd.DataFrame(history)
        history_df.to_csv(outdir / "training_history.csv", index=False)

    def export_predictions(self, predictions: np.ndarray, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, predictions)

    def export_weights_to_txt(self, model: EvalNet, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        sd = model.state_dict()

        W1 = sd["fc1.weight"].cpu().numpy()
        b1 = sd["fc1.bias"].cpu().numpy()
        W2 = sd["fc2.weight"].cpu().numpy()
        b2 = sd["fc2.bias"].cpu().numpy()

        with open(path, "w") as f:
            for arr in [W1.flatten(), b1.flatten(), W2.flatten(), b2.flatten()]:
                for v in arr:
                    f.write(f"{float(v):.10f}\n")

    def export_weights_to_header(self, model: EvalNet, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        sd = model.state_dict()

        W1 = sd["fc1.weight"].cpu().numpy()
        b1 = sd["fc1.bias"].cpu().numpy()
        W2 = sd["fc2.weight"].cpu().numpy()
        b2 = sd["fc2.bias"].cpu().numpy()

        input_size = self.cfg["model"]["input_size"]
        hidden_size = self.cfg["model"]["hidden_size"]

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
        content.append(f"inline constexpr int INPUT_SIZE = {input_size};\n")
        content.append(f"inline constexpr int HIDDEN_SIZE = {hidden_size};\n\n")
        content.append(arr2("W1", W1))
        content.append("\n")
        content.append(arr1("b1", b1))
        content.append("\n")
        content.append(arr2("W2", W2))
        content.append("\n")
        content.append(arr1("b2", b2))
        content.append("\n")
        content.append("} // namespace nn_weights\n")

        with open(path, "w") as f:
            f.write("".join(content))
