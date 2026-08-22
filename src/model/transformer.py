import torch
from torch import nn

from model.attention import MultiHeadAttention


class LayerNorm(nn.Module):
    def __init__(self, model_dimension: int, epsilon: float = 1e-5) -> None:
        super().__init__()

        if model_dimension <= 0:
            raise ValueError("model_dimension must be positive")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        self.model_dimension = model_dimension
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(model_dimension))
        self.beta = nn.Parameter(torch.zeros(model_dimension))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "x must have shape (batch_size, sequence_length, model_dimension)"
            )
        if x.size(-1) != self.model_dimension:
            raise ValueError(
                "the final dimension of x must match model_dimension "
                f"({x.size(-1)} != {self.model_dimension})"
            )

        mean = x.mean(dim=-1, keepdim=True)
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + self.epsilon)

        return self.gamma * normalized + self.beta


class FeedForward(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        expansion_factor: int = 4,
    ) -> None:
        super().__init__()

        if model_dimension <= 0:
            raise ValueError("model_dimension must be positive")
        if expansion_factor <= 0:
            raise ValueError("expansion_factor must be positive")

        self.model_dimension = model_dimension
        hidden_dimension = expansion_factor * model_dimension
        self.layers = nn.Sequential(
            nn.Linear(model_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, model_dimension),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "x must have shape (batch_size, sequence_length, model_dimension)"
            )
        if x.size(-1) != self.model_dimension:
            raise ValueError(
                "the final dimension of x must match model_dimension "
                f"({x.size(-1)} != {self.model_dimension})"
            )

        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        num_of_heads: int,
        model_dimension: int,
        dropout: float = 0.0,
        ff_expansion_factor: int = 4,
    ) -> None:
        super().__init__()

        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1")

        self.norm_before_attention = LayerNorm(model_dimension)
        self.attention = MultiHeadAttention(
            num_of_heads=num_of_heads,
            model_dimension=model_dimension,
            dropout=dropout,
            causal=True,
        )
        self.norm_before_feed_forward = LayerNorm(model_dimension)
        self.feed_forward = FeedForward(
            model_dimension=model_dimension,
            expansion_factor=ff_expansion_factor,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention_output = self.attention(
            self.norm_before_attention(x),
            attention_mask=attention_mask,
        )
        x = x + self.dropout(attention_output)

        feed_forward_output = self.feed_forward(self.norm_before_feed_forward(x))
        return x + self.dropout(feed_forward_output)
