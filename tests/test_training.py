import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from generation import generate_tokens
from model.gpt import GPTModel
from training import compute_language_model_loss, train_model


class TrainingUtilityTests(unittest.TestCase):
    def test_language_model_loss_is_a_differentiable_scalar(self):
        logits = torch.randn(2, 3, 5, requires_grad=True)
        targets = torch.randint(5, (2, 3), dtype=torch.long)

        loss = compute_language_model_loss(logits, targets)
        loss.backward()

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)

    def test_generation_appends_requested_number_of_tokens(self):
        model = GPTModel(
            vocab_size=20,
            model_dimension=4,
            max_context_length=4,
            num_of_heads=2,
            num_of_layers=1,
        )
        prompt_ids = torch.tensor([[1, 2]], dtype=torch.long)

        generated_ids = generate_tokens(
            model=model,
            token_ids=prompt_ids,
            max_new_tokens=3,
            context_length=4,
            top_k=5,
            allowed_token_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        )

        self.assertEqual(generated_ids.shape, (1, 5))
        self.assertTrue(
            torch.all((generated_ids[:, :2] >= 0) & (generated_ids[:, :2] < 20))
        )
        self.assertTrue(
            torch.all((generated_ids[:, 2:] >= 0) & (generated_ids[:, 2:] <= 2))
        )

    def test_training_evaluates_at_the_end_of_a_short_epoch(self):
        model = GPTModel(
            vocab_size=10,
            model_dimension=4,
            max_context_length=4,
            num_of_heads=2,
            num_of_layers=1,
        )
        input_ids = torch.randint(10, (2, 3), dtype=torch.long)
        target_ids = torch.randint(10, (2, 3), dtype=torch.long)
        data_loader = DataLoader(TensorDataset(input_ids, target_ids), batch_size=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        history = train_model(
            model=model,
            train_loader=data_loader,
            validation_loader=data_loader,
            optimizer=optimizer,
            device=torch.device("cpu"),
            num_epochs=1,
            evaluation_interval=100,
            evaluation_batches=1,
        )

        self.assertEqual(len(history.evaluations), 1)
