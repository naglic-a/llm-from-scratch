import torch
from torch import nn

from model.embeddings import TokenAndPositionEmbedding
from model.transformer import LayerNorm, TransformerBlock


class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dimension: int,
        max_context_length: int,
        num_of_heads: int,
        num_of_layers: int,
        dropout: float = 0.0,
        ff_expansion_factor: int = 4,
    ) -> None:
        super().__init__()

        if num_of_layers <= 0:
            raise ValueError("num_of_layers must be positive")

        self.embedding = TokenAndPositionEmbedding(
            vocab_size=vocab_size,
            model_dimension=model_dimension,
            max_context_length=max_context_length,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    num_of_heads=num_of_heads,
                    model_dimension=model_dimension,
                    dropout=dropout,
                    ff_expansion_factor=ff_expansion_factor,
                )
                for _ in range(num_of_layers)
            ]
        )
        self.final_norm = LayerNorm(model_dimension)
        self.output_projection = nn.Linear(model_dimension, vocab_size, bias=False)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embedding(token_ids)

        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        x = self.final_norm(x)
        return self.output_projection(x)
