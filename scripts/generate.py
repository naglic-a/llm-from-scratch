from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import ModelConfig
from data.tokenizer import load_tokenizer
from generation import generate_tokens
from model.gpt import GPTModel
from training import resolve_device


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def standalone_decodable_token_ids(tokenizer: object) -> list[int]:
    valid_ids: list[int] = []
    for token_id in range(tokenizer.vocab_size()):
        try:
            tokenizer.decode([token_id])
        except UnicodeError:
            continue
        valid_ids.append(token_id)

    if not valid_ids:
        raise ValueError("tokenizer has no independently decodable tokens")

    return valid_ids


def main() -> None:
    arguments = parse_arguments()
    device = resolve_device(arguments.device)
    checkpoint = torch.load(
        arguments.checkpoint,
        map_location=device,
        weights_only=True,
    )

    model_config = ModelConfig(**checkpoint["model_config"])
    model = GPTModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    tokenizer = load_tokenizer(checkpoint["tokenizer_path"])
    prompt_ids = tokenizer.encode(arguments.prompt)
    if not prompt_ids:
        raise ValueError("prompt must encode to at least one token")

    token_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    allowed_token_ids = torch.tensor(
        standalone_decodable_token_ids(tokenizer),
        dtype=torch.long,
        device=device,
    )
    generated_ids = generate_tokens(
        model=model,
        token_ids=token_ids,
        max_new_tokens=arguments.max_new_tokens,
        context_length=model_config.max_context_length,
        temperature=arguments.temperature,
        top_k=arguments.top_k,
        allowed_token_ids=allowed_token_ids,
    )
    print(tokenizer.decode(generated_ids[0].tolist()))


if __name__ == "__main__":
    main()
