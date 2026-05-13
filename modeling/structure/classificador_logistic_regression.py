import torch
import torch.nn as nn
import numpy as np

from torch import device as TorchDevice
from torch.utils.data import TensorDataset, DataLoader

from tracker.base_tracker import BaseTracker
from utils.experiment_io import get_run_dir


class LogisticRegressionClassifier(nn.Module):
    def __init__(self, config, logger, device: TorchDevice, tracker: BaseTracker):
        super().__init__()
        self.config = config
        self.logger = logger
        self.tracker = tracker
        self.device = device

        self.phase = self.config.get("phase")
        self.model_dir = (
            get_run_dir(self.config.get("run_id")) / f"{self.phase}_model.pt"
        )

        self.learning_rate = (
            self.config.get("modeling", {}).get("training", {}).get("learning_rate")
        )
        self.num_epochs = (
            self.config.get("modeling", {}).get("training", {}).get("num_epochs")
        )
        self.batch_size = (
            self.config.get("modeling", {}).get("training", {}).get("batch_size")
        )

        self.input_dim = (
            config.get("modeling", {}).get("structure", {}).get("input_dim")
        )
        self.num_classes = (
            config.get("modeling", {}).get("structure", {}).get("num_classes")
        )

        # Modelo linear:
        # y = Wx + b
        self.linear = nn.Linear(self.input_dim, self.num_classes)

        self.optimizer = None
        self.criterion = None

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

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.float()

        logits = self.linear(x)

        return logits

    def fit(
        self,
        X: np.ndarray,
        y,
    ):
        X_tensor = torch.from_numpy(X).float().to(self.device)

        # Compatível com pandas categorical
        if hasattr(y, "codes"):
            y_tensor = torch.tensor(y.codes, dtype=torch.long, device=self.device)
        else:
            y_tensor = torch.tensor(y, dtype=torch.long, device=self.device)

        self.num_classes = torch.unique(y_tensor)

        dataset = TensorDataset(X_tensor, y_tensor)

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.train()

        for epoch in range(self.num_epochs):

            epoch_loss = 0.0

            for batch_x, batch_y in loader:

                self.optimizer.zero_grad()

                logits = self.forward(batch_x)

                loss = self.criterion(logits, batch_y)

                loss.backward()

                self.optimizer.step()

                epoch_loss += loss.item()

                self.tracker.log_metrics({"epoch": epoch, "loss": loss.item()})

            self.logger.info(
                f"Epoch [{epoch+1}/{self.num_epochs}] "
                f"Loss: {epoch_loss/len(loader):.6f}"
            )

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> torch.Tensor:

        self.eval()

        X_tensor = torch.from_numpy(X).float().to(self.device)

        logits = self.forward(X_tensor)

        predictions = torch.argmax(logits, dim=1)

        return predictions

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> torch.Tensor:

        self.eval()

        X_tensor = torch.from_numpy(X).float().to(self.device)

        logits = self.forward(X_tensor)

        probabilities = torch.softmax(logits, dim=1)

        return probabilities
