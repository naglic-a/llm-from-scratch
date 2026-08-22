from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.nn import functional as functional
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EvaluationMetrics:
    epoch: int
    step: int
    train_loss: float
    validation_loss: float


@dataclass
class TrainingHistory:
    evaluations: list[EvaluationMetrics] = field(default_factory=list)


def resolve_device(requested_device: str = "auto") -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def compute_language_model_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute next-token cross-entropy for logits [B, T, V] and targets [B, T]."""
    if logits.ndim != 3:
        raise ValueError(
            "logits must have shape (batch_size, sequence_length, vocab_size)"
        )
    if target_ids.ndim != 2:
        raise ValueError("target_ids must have shape (batch_size, sequence_length)")
    if logits.shape[:2] != target_ids.shape:
        raise ValueError("logits and target_ids must agree on batch and sequence axes")
    if target_ids.dtype != torch.long:
        raise TypeError("target_ids must have torch.long dtype")

    return functional.cross_entropy(logits.flatten(0, 1), target_ids.flatten())


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided")

    was_training = model.training
    model.eval()
    total_loss = 0.0
    batch_count = 0

    try:
        for input_ids, target_ids in data_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            logits = model(input_ids)
            total_loss += compute_language_model_loss(logits, target_ids).item()
            batch_count += 1

            if max_batches is not None and batch_count >= max_batches:
                break
    finally:
        model.train(was_training)

    if batch_count == 0:
        raise ValueError("data_loader produced no evaluation batches")

    return total_loss / batch_count


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    evaluation_interval: int,
    evaluation_batches: int,
    max_gradient_norm: float | None = None,
    on_evaluation: Callable[[EvaluationMetrics], None] | None = None,
) -> TrainingHistory:
    """Optimize a language model and periodically record train/validation loss."""
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    if evaluation_interval <= 0:
        raise ValueError("evaluation_interval must be positive")
    if evaluation_batches <= 0:
        raise ValueError("evaluation_batches must be positive")
    if max_gradient_norm is not None and max_gradient_norm <= 0:
        raise ValueError("max_gradient_norm must be positive when provided")

    model.to(device)
    history = TrainingHistory()
    global_step = 0
    last_evaluation_step = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        last_training_loss: float | None = None
        for input_ids, target_ids in train_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids)
            loss = compute_language_model_loss(logits, target_ids)
            loss.backward()

            if max_gradient_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)

            optimizer.step()
            global_step += 1
            last_training_loss = loss.item()

            if global_step % evaluation_interval != 0:
                continue

            metrics = EvaluationMetrics(
                epoch=epoch,
                step=global_step,
                train_loss=loss.item(),
                validation_loss=evaluate_loss(
                    model,
                    validation_loader,
                    device,
                    max_batches=evaluation_batches,
                ),
            )
            history.evaluations.append(metrics)
            last_evaluation_step = global_step
            if on_evaluation is not None:
                on_evaluation(metrics)

        if last_training_loss is None:
            raise ValueError("train_loader produced no training batches")
        if global_step == last_evaluation_step:
            continue

        metrics = EvaluationMetrics(
            epoch=epoch,
            step=global_step,
            train_loss=last_training_loss,
            validation_loss=evaluate_loss(
                model,
                validation_loader,
                device,
                max_batches=evaluation_batches,
            ),
        )
        history.evaluations.append(metrics)
        last_evaluation_step = global_step
        if on_evaluation is not None:
            on_evaluation(metrics)

    return history
