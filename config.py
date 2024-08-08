from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Parameter defaults for a training run obtained from hyperparameter optimization."""

    # batch_size: int = 128
    learning_rate: float = 0.00005
    # learning_rate: float = 0.000005
    weight_decay: float = 0
    sgd_momentum: float = 0.8
    scheduler_gamma: float = 0.8
    pos_weight: float = 1
    embedding_size: int = 64
    n_heads: int = 3
    n_layers: int = 3
    dropout_rate: float = 0.2
    top_k_ratio: float = 0.5
    top_k_every_n: int = 1
    dense_neurons: int = 256
