import torch

from model.gpt import GPTModel


@torch.no_grad()
def generate_tokens(
    model: GPTModel,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    allowed_token_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Append sampled token IDs to one batch of prompt token IDs."""
    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape (batch_size, sequence_length)")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when provided")
    if allowed_token_ids is not None:
        if allowed_token_ids.ndim != 1:
            raise ValueError("allowed_token_ids must be one-dimensional")
        if allowed_token_ids.numel() == 0:
            raise ValueError("allowed_token_ids cannot be empty")
        if allowed_token_ids.dtype != torch.long:
            raise TypeError("allowed_token_ids must have torch.long dtype")

    was_training = model.training
    model.eval()

    try:
        for _ in range(max_new_tokens):
            context = token_ids[:, -context_length:]
            next_token_logits = model(context)[:, -1, :] / temperature

            if allowed_token_ids is not None:
                allowed_ids = allowed_token_ids.to(next_token_logits.device)
                if torch.any(allowed_ids < 0) or torch.any(
                    allowed_ids >= next_token_logits.size(-1)
                ):
                    raise ValueError(
                        "allowed_token_ids contains an out-of-range token ID"
                    )
                allowed_mask = torch.ones_like(next_token_logits, dtype=torch.bool)
                allowed_mask[:, allowed_ids] = False
                next_token_logits = next_token_logits.masked_fill(
                    allowed_mask,
                    torch.finfo(next_token_logits.dtype).min,
                )

            if top_k is not None:
                effective_top_k = min(top_k, next_token_logits.size(-1))
                threshold = torch.topk(
                    next_token_logits,
                    effective_top_k,
                ).values[:, -1:]
                next_token_logits = next_token_logits.masked_fill(
                    next_token_logits < threshold,
                    torch.finfo(next_token_logits.dtype).min,
                )

            probabilities = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim=1)
    finally:
        model.train(was_training)

    return token_ids
