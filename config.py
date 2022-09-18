from dataclasses import dataclass
from hydra.core.config_store import ConfigStore


@dataclass
class Paths:
    log: str
    data: str


@dataclass
class Files:
    train_data: str
    train_labels: str
    test_data: str
    test_labels: str


@dataclass
class Params:
    epoch_count: int
    lr: float
    batch_size: int


@dataclass
class MNISTConfig:
    paths: Paths
    files: Files
    params: Params


cs = ConfigStore.instance()
cs.store(name="mnist_config", node=MNISTConfig)
