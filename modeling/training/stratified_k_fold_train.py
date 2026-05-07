import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from modeling.training.early_stopping import EarlyStopping
from modeling.training.pytorch_base import PytorchTrainingAlgorithm
from modeling.structure.pytorch_base import PytorchModelStructure

from utils.create_loader import create_loader
from utils.criterion import get_criterion
from utils.seed_all import DEFAULT_SEED


class StratifiedKFoldTrain(PytorchTrainingAlgorithm):
    def fit(
        self,
        model: PytorchModelStructure,
        train_loader: DataLoader,
        epoch: int,
        fold_id: int
    ) -> float:

        model.train()
        train_loss = 0
        y_true = []
        y_prob = []

        for data, target, _ in train_loader:
            self.optimizer.zero_grad()
            logits = model.forward(data)
            loss = self.criterion(logits, target)
            loss.backward()

            train_loss += loss.item()

            self.optimizer.step()

            probs = torch.softmax(logits, dim=1)
            y_prob.extend(probs.cpu().detach().numpy())
            y_true.extend(target.cpu().numpy())

        train_loss = train_loss / len(train_loader)

        self.logger.info(f'fold_id: {fold_id} \tEpoch: {epoch} \tTraining loss: {train_loss:.6f}')

        return (
            np.array(y_true),
            np.array(y_prob),
            train_loss
        )

    def validate(
        self,
        model: PytorchModelStructure,
        val_loader: DataLoader,
        epoch: int,
        fold_id: int
    ) -> float:

        model.eval()
        val_loss = 0
        y_true = []
        y_prob = []

        with torch.no_grad():
            for data, target, _ in val_loader:
                logits = model.forward(data)
                loss = self.criterion(logits, target)
                val_loss += loss.item()

                probs = torch.softmax(logits, dim=1)
                y_prob.extend(probs.cpu().numpy())
                y_true.extend(target.cpu().numpy())

        val_loss = val_loss / len(val_loader)
        self.logger.info(f'fold_id: {fold_id} \tEpoch: {epoch} \tValidation loss: {val_loss:.6f}')

        return (
            np.array(y_true),
            np.array(y_prob),
            val_loss
        )

    def __create_loaders(
        self,
        X,
        y,
        train_idx,
        val_idx
    ):

        data = [
            [X[i], y.iloc[i]["label"], i]
            for i in range(len(X))
        ]

        g = torch.Generator()
        g.manual_seed(DEFAULT_SEED)

        batch_size = (
            self.config
            .get("modeling", {})
            .get("training", {})
            .get("batch_size", 32)
        )

        train_loader = create_loader(
            data,
            train_idx,
            batch_size,
            self.device,
            g
        )

        val_loader = create_loader(
            data,
            val_idx,
            batch_size,
            self.device,
            g
        )

        return train_loader, val_loader
    
    def train_epoch(self, fold_id, X, y, train_idx, test_idx, model, num_epochs):
        self.logger.info(f"Starting Fold {fold_id + 1}")

        train_loader, val_loader = self.__create_loaders(X, y, train_idx, test_idx)

        # reinicializa pesos
        model.reset_weights()

        for epoch in range(num_epochs):
            train_out = self.fit(model, train_loader, epoch, fold_id)
            val_out = self.validate(model, val_loader, epoch, fold_id)

            # self.tracker.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
            model.save_model_state_dict(fold_id=fold_id)

        return train_out, val_out

    def train(
        self,
        model: PytorchModelStructure,
        X: np.ndarray,
        y: pd.DataFrame
    ):

        criterion_name  =   self.config.get('modeling', {}).get('training', {}).get('criterion')
        reduction       =   self.config.get('modeling', {}).get('training', {}).get('reduction', 'mean')
        learning_rate   =   self.config.get('modeling', {}).get('training', {}).get('learning_rate')
        num_epochs      =   self.config.get('modeling', {}).get('training', {}).get('num_epochs')


        self.optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
        self.criterion = get_criterion(criterion_name, reduction=reduction)

        y_labels = y["label"].values

        splitter = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=DEFAULT_SEED)

        fold_train_losses = []
        fold_val_losses = []
        for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y_labels)):
            if self.config.get('phase') == 'train':
                inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                X_train_outer = X[train_idx]
                y_train_outer = y.iloc[train_idx].reset_index(drop=True)

                for inner_fold_id, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_train_outer, y_train_outer['label'])):
                    train_loss, fold_val_loss = self.train_epoch(inner_fold_id, X, y, inner_train_idx, inner_val_idx, model, num_epochs)
                    fold_train_losses.append(train_loss)
                    fold_val_losses.append(fold_val_loss)

                return fold_train_losses, fold_val_losses
            
            train_loss, fold_val_loss = self.train_epoch(fold_id, X, y, train_idx, test_idx, model, num_epochs)
            fold_train_losses.append(train_loss)
            fold_val_losses.append(fold_val_loss)

        return fold_train_losses, fold_val_losses

    @staticmethod
    def reset_weights(m):
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()