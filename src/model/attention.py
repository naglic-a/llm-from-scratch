import math

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, num_of_heads, model_dimension, dropout=0.0, causal=False):
        super().__init__()

        if num_of_heads <= 0:
            raise ValueError("num_of_heads must be positive")
        if model_dimension <= 0:
            raise ValueError("model_dimension must be positive")
        if model_dimension % num_of_heads != 0:
            raise ValueError(
                "model_dimension must be divisible by num_of_heads "
                f"(got {model_dimension} and {num_of_heads})"
            )

        self.n_head = num_of_heads
        self.d_model = model_dimension
        self.d_head = model_dimension // num_of_heads
        self.causal = causal

        self.query_projection = nn.Linear(model_dimension, model_dimension)
        self.key_projection = nn.Linear(model_dimension, model_dimension)
        self.value_projection = nn.Linear(model_dimension, model_dimension)
        self.output_projection = nn.Linear(model_dimension, model_dimension)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x): # private helper method
        batch_size, sequence_length, _ = x.shape

        x = x.view(
            batch_size,
            sequence_length,
            self.n_head,
            self.d_head,
        )
        return x.transpose(1, 2)

    def forward(self, x, attention_mask=None):
        if x.ndim != 3:
            raise ValueError(
                "x must have shape (batch_size, sequence_length, model_dimension)"
            )
        if x.size(-1) != self.d_model:
            raise ValueError(
                f"the final dimension of x must be {self.d_model}, got {x.size(-1)}"
            )

        query = self._split_heads(self.query_projection(x))
        key = self._split_heads(self.key_projection(x))
        value = self._split_heads(self.value_projection(x))

        # (batch, heads, query_length, key_length)
        scores = query @ key.transpose(-2, -1)
        scores = scores / math.sqrt(self.d_head)

        if self.causal:
            sequence_length = x.size(1)
            causal_mask = torch.triu(
                torch.ones(
                    sequence_length,
                    sequence_length,
                    device=x.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)

        if attention_mask is not None:
            if attention_mask.dtype != torch.bool:
                raise TypeError("attention_mask must have boolean dtype")
            scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = weights @ value

        # (batch, heads, sequence_length, d_head)
        context = context.transpose(1, 2).contiguous()
        context = context.view(x.size(0), x.size(1), self.d_model)

        return self.output_projection(context)
