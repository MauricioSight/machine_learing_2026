import torch
import torch.nn as nn
import numpy as np

from torch import device as TorchDevice
from torch.utils.data import TensorDataset, DataLoader

from tracker.base_tracker import BaseTracker
from utils.experiment_io import get_run_dir


class LogisticRegressionClassifier(nn.Module):
    def __init__(
        self,
        config,
        logger,
        device: TorchDevice,
        tracker: BaseTracker,
    ):
        super().__init__()

        self.config = config
        self.logger = logger
        self.tracker = tracker
        self.device = device

        self.phase = self.config.get("phase")
        self.model_dir = (
            get_run_dir(self.config.get("run_id")) / f"{self.phase}_model.pt"
        )

        training_cfg = self.config.get("modeling", {}).get("training", {})

        self.learning_rate = training_cfg.get("learning_rate", 1e-3)
        self.num_epochs = training_cfg.get("num_epochs", 100)
        self.batch_size = training_cfg.get("batch_size", 32)
        self.weight_decay = training_cfg.get("weight_decay", 0.0)
        self.optimizer_name = training_cfg.get("optimizer", "adam")
        self.patience = training_cfg.get("patience", 20)
        self.gradient_clip = training_cfg.get("gradient_clip", None)
        structure_cfg = self.config.get("modeling", {}).get("structure", {})

        self.input_dim = structure_cfg.get("input_dim")
        self.num_classes = structure_cfg.get("num_classes")

        self.linear = nn.Linear(self.input_dim, self.num_classes)

        self.optimizer = None
        self.criterion = None
        self.scheduler = None

        self.to(self.device)

    def save(self):
        torch.save(
            {
                "model_state_dict": self.state_dict(),
            },
            self.model_dir,
        )

    def load(self):
        checkpoint = torch.load(self.model_dir, map_location=self.device)

        self.load_state_dict(checkpoint["model_state_dict"])

        self.to(self.device)

        self.eval()

    def reset_weights(self):
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def compile(self):
        self.criterion = nn.CrossEntropyLoss()

        optimizer_name = self.optimizer_name.lower()

        if optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        elif optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        elif optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
            )

        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = x.float()

        logits = self.linear(x)

        return logits

    def fit(
        self,
        X_train: np.ndarray,
        y_train,
        X_val: np.ndarray = None,
        y_val=None,
        verbose=True,
    ):
        X_tensor = torch.from_numpy(X_train).float().to(self.device)

        if hasattr(y_train, "codes"):
            y_tensor = torch.tensor(
                y_train.codes,
                dtype=torch.long,
                device=self.device,
            )

        else:
            y_tensor = torch.tensor(
                y_train,
                dtype=torch.long,
                device=self.device,
            )

        self.num_classes = len(torch.unique(y_tensor))

        dataset = TensorDataset(
            X_tensor,
            y_tensor,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        use_validation = X_val is not None and y_val is not None

        if use_validation:
            X_val_tensor = torch.from_numpy(X_val).float().to(self.device)

            if hasattr(y_val, "codes"):
                y_val_tensor = torch.tensor(
                    y_val.codes,
                    dtype=torch.long,
                    device=self.device,
                )

            else:
                y_val_tensor = torch.tensor(
                    y_val,
                    dtype=torch.long,
                    device=self.device,
                )

        self.train()

        best_val_loss = float("inf")
        patience_counter = 0

        best_state_dict = None

        num_logs = 10
        log_every = max(1, self.num_epochs // num_logs)

        for epoch in range(self.num_epochs):
            self.train()

            epoch_loss = 0.0

            for batch_x, batch_y in loader:

                self.optimizer.zero_grad()

                logits = self.forward(batch_x)

                loss = self.criterion(logits, batch_y)

                loss.backward()

                # Gradient clipping
                if self.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(),
                        self.gradient_clip,
                    )

                self.optimizer.step()

                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(loader)

            val_loss = None

            if use_validation:
                self.eval()

                with torch.no_grad():
                    val_logits = self.forward(X_val_tensor)
                    val_loss = self.criterion(val_logits, y_val_tensor).item()

                self.scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state_dict = {
                        k: v.cpu().clone() for k, v in self.state_dict().items()
                    }
                    patience_counter = 0

                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if verbose:
                            self.logger.info(f"Early stopping at epoch {epoch+1}")

                        break

            metrics = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
            }

            if val_loss is not None:
                metrics["val_loss"] = val_loss

            if self.tracker is not None:
                self.tracker.log_metrics(metrics)

            should_log = (
                epoch == 0 or epoch == self.num_epochs - 1 or epoch % log_every == 0
            )
            if should_log and verbose:
                log_msg = (
                    f"Epoch [{epoch+1}/{self.num_epochs}] "
                    f"Train Loss: {avg_train_loss:.6f}"
                )
                if val_loss is not None:
                    log_msg += f" | Val Loss: {val_loss:.6f}"

                self.logger.info(log_msg)

        # Restore best model
        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)

    @torch.no_grad()
    def predict(
        self,
        X: np.ndarray,
    ) -> torch.Tensor:

        self.eval()

        probs = self.predict_proba(X)

        predictions = torch.argmax(probs, dim=1)

        return predictions

    @torch.no_grad()
    def predict_proba(
        self,
        X: np.ndarray,
    ) -> torch.Tensor:

        self.eval()

        X_tensor = torch.from_numpy(X).float().to(self.device)

        logits = self.forward(X_tensor)

        probabilities = torch.softmax(logits, dim=1)

        return probabilities
