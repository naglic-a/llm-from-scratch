from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    model_dimension: int = 128
    max_context_length: int = 64
    num_of_heads: int = 4
    num_of_layers: int = 2
    dropout: float = 0.1
    ff_expansion_factor: int = 4

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.model_dimension <= 0:
            raise ValueError("model_dimension must be positive")
        if self.max_context_length <= 0:
            raise ValueError("max_context_length must be positive")
        if self.num_of_heads <= 0:
            raise ValueError("num_of_heads must be positive")
        if self.model_dimension % self.num_of_heads != 0:
            raise ValueError("model_dimension must be divisible by num_of_heads")
        if self.num_of_layers <= 0:
            raise ValueError("num_of_layers must be positive")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1")
        if self.ff_expansion_factor <= 0:
            raise ValueError("ff_expansion_factor must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 8
    context_length: int = 64
    stride: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    evaluation_interval: int = 100
    evaluation_batches: int = 10
    max_gradient_norm: float | None = 1.0

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.evaluation_interval <= 0:
            raise ValueError("evaluation_interval must be positive")
        if self.evaluation_batches <= 0:
            raise ValueError("evaluation_batches must be positive")
        if self.max_gradient_norm is not None and self.max_gradient_norm <= 0:
            raise ValueError("max_gradient_norm must be positive when provided")
