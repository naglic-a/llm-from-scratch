import torch
from torch import nn


class TokenAndPositionEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dimension: int,
        max_context_length: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if model_dimension <= 0:
            raise ValueError("model_dimension must be positive")
        if max_context_length <= 0:
            raise ValueError("max_context_length must be positive")

        self.model_dimension = model_dimension
        self.max_context_length = max_context_length

        self.token_embedding = nn.Embedding(vocab_size, model_dimension)
        self.position_embedding = nn.Embedding(
            max_context_length,
            model_dimension,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError(
                "token_ids must have shape "
                "(batch_size, sequence_length)"
            )
        if token_ids.dtype != torch.long:
            raise TypeError("token_ids must have torch.long dtype")

        sequence_length = token_ids.size(1)
        if sequence_length > self.max_context_length:
            raise ValueError(
                "sequence_length cannot exceed max_context_length "
                f"({sequence_length} > {self.max_context_length})"
            )

        positions = torch.arange(
            sequence_length,
            device=token_ids.device,
            dtype=torch.long,
        )

        tokens = self.token_embedding(token_ids)
        positions = self.position_embedding(positions)

        return self.dropout(tokens + positions)
