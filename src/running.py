import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from typing import Any, Optional
from src.tracking import ExperimentTracker, Stage
from src.metrics import Metric
from tqdm import tqdm
from sklearn.metrics import accuracy_score


class Runner:
    def __init__(
        self,
        loader: DataLoader[Any],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
    ):
        self.run_count = 0
        self.loader = loader
        self.model = model
        self.optimizer = optimizer
        self.accuracy_metric = Metric()
        self.compute_loss = torch.nn.CrossEntropyLoss(reduction="mean")
        self.y_true_batches: list[list[Any]] = []
        self.y_pred_batches: list[list[Any]] = []
        self.stage = Stage.VAL if optimizer is None else Stage.TRAIN

    @property
    def avg_accuracy(self):
        return self.accuracy_metric.average

    def run(self, desc: str, experiment: ExperimentTracker):
        for x, y in tqdm(self.loader, desc=desc, ncols=80):
            batch_accuracy = self._run_epoch(x, y)
            experiment.add_batch_metric("accuracy", batch_accuracy, self.run_count)

    def _run_epoch(self, x, y):
        self.run_count += 1
        batch_size = x.shape[0]
        predicted = self.model(x)
        loss = self.compute_loss(predicted, y)

        # compute Batch Metrics
        y_np = y.detach().numpy()
        y_prediction_np = np.argmax(predicted.detach().numpy(), axis=1)
        batch_accuracy: float = accuracy_score(y_np, y_prediction_np)
        self.accuracy_metric.update(batch_accuracy, batch_size)

        self.y_true_batches += [y_np]
        self.y_pred_batches += [y_prediction_np]

        if self.optimizer:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return batch_accuracy

    def reset(self):
        self.accuracy_metric = Metric()
        self.y_true_batches = []
        self.y_pred_batches = []


def run_epoch(
    test_runner: Runner,
    train_runner: Runner,
    experiment: ExperimentTracker,
    epoch_id: int,
    epoch_total: int,
):
    # Testing Loop
    experiment.set_stage(Stage.TRAIN)
    train_runner.run(f"Training Epoch {epoch_id}", experiment)
    # log training epoch metrics
    experiment.add_epoch_metric("accuracy", train_runner.avg_accuracy, epoch_id)

    experiment.set_stage(Stage.VAL)
    test_runner.run(f"Validation Epoch {epoch_id}", experiment)
    # log validation epoch metrics
    experiment.add_epoch_metric("accuracy", test_runner.avg_accuracy, epoch_id)

    experiment.add_epoch_confusion_matrix(
        test_runner.y_true_batches, test_runner.y_pred_batches, epoch_id
    )

    # Compute Average Epoch Metrics
    summary = ", ".join(
        [
            f"[Epoch: {epoch_id + 1}/{epoch_total}]",
            f"Test Accuracy: {test_runner.avg_accuracy: 0.4f}",
            f"Train Accuracy: {train_runner.avg_accuracy: 0.4f}",
        ]
    )
    print("\n" + summary + "\n")

    # Reset metrics
    test_runner.reset()
    train_runner.reset()
