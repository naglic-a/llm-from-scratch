from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from config import ModelConfig, TrainingConfig
from data.dataset import createDataLoader
from data.tokenizer import train_or_load_tokenizer
from model.gpt import GPTModel
from training import EvaluationMetrics, resolve_device, train_model


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", type=Path, required=True, help="UTF-8 corpus path")
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path("artifacts/tokenizer.json"),
        help="saved Rust BPE tokenizer path",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/gpt.pt"),
        help="destination for the final checkpoint",
    )
    parser.add_argument("--vocab-size", type=int, default=2_000)
    parser.add_argument("--model-dimension", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint if it exists")
    return parser.parse_args()


def split_train_validation_text(
    text: str,
    validation_fraction: float = 0.1,
) -> tuple[str, str]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")

    split_index = int(len(text) * (1.0 - validation_fraction))
    train_text = text[:split_index]
    validation_text = text[split_index:]

    if not train_text or not validation_text:
        raise ValueError("text is too short to create train and validation splits")

    return train_text, validation_text


def main() -> None:
    arguments = parse_arguments()
    torch.manual_seed(arguments.seed)

    text = arguments.text.read_text(encoding="utf-8")
    train_text, validation_text = split_train_validation_text(text)

    training_config = TrainingConfig(
        batch_size=arguments.batch_size,
        context_length=arguments.context_length,
        stride=arguments.context_length,
        learning_rate=arguments.learning_rate,
        num_epochs=arguments.num_epochs,
    )
    print("Loading tokenizer...")
    tokenizer = train_or_load_tokenizer(
        texts=[train_text],
        path=arguments.tokenizer_path,
        target_vocab_size=arguments.vocab_size,
    )
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size(),
        model_dimension=arguments.model_dimension,
        max_context_length=training_config.context_length,
        num_of_heads=arguments.num_heads,
        num_of_layers=arguments.num_layers,
        dropout=arguments.dropout,
    )

    print("Encoding train dataset...")
    train_loader = createDataLoader(
        train_text,
        tokenizer,
        batch_size=training_config.batch_size,
        max_length=training_config.context_length,
        stride=training_config.stride,
        shuffle=True,
        drop_last=True,
    )
    print("Encoding validation dataset...")
    validation_loader = createDataLoader(
        validation_text,
        tokenizer,
        batch_size=training_config.batch_size,
        max_length=training_config.context_length,
        stride=training_config.stride,
        shuffle=False,
        drop_last=False,
    )

    device = resolve_device(arguments.device)
    model = GPTModel(**asdict(model_config))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    evaluations = []
    if arguments.resume and arguments.checkpoint.exists():
        print(f"Resuming training from {arguments.checkpoint}...")
        checkpoint_data = torch.load(arguments.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        if "evaluations" in checkpoint_data:
            evaluations = checkpoint_data["evaluations"]

    def log_evaluation(metrics: EvaluationMetrics) -> None:
        print(
            f"epoch={metrics.epoch} step={metrics.step} "
            f"train_loss={metrics.train_loss:.4f} "
            f"validation_loss={metrics.validation_loss:.4f}"
        )

    print(f"training on {device} with vocabulary size {model_config.vocab_size}")
    
    try:
        history = train_model(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=training_config.num_epochs,
            evaluation_interval=training_config.evaluation_interval,
            evaluation_batches=training_config.evaluation_batches,
            max_gradient_norm=training_config.max_gradient_norm,
            on_evaluation=log_evaluation,
        )
        evaluations.extend([asdict(metrics) for metrics in history.evaluations])
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)! Saving current progress...")

    arguments.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "tokenizer_path": str(arguments.tokenizer_path),
            "evaluations": evaluations,
        },
        arguments.checkpoint,
    )
    print(f"saved checkpoint to {arguments.checkpoint}")


if __name__ == "__main__":
    main()
