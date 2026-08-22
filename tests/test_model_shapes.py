import unittest

import torch

from model.attention import MultiHeadAttention
from model.embeddings import TokenAndPositionEmbedding
from model.gpt import GPTModel
from model.transformer import FeedForward, LayerNorm, TransformerBlock


class ModelShapeTests(unittest.TestCase):
    def test_embedding_output_shape(self):
        embedding = TokenAndPositionEmbedding(
            vocab_size=500,
            model_dimension=128,
            max_context_length=32,
        )
        token_ids = torch.randint(500, (2, 8), dtype=torch.long)

        output = embedding(token_ids)

        self.assertEqual(output.shape, (2, 8, 128))

    def test_attention_preserves_model_shape(self):
        attention = MultiHeadAttention(
            num_of_heads=4,
            model_dimension=128,
            causal=True,
        )
        embeddings = torch.randn(2, 8, 128)

        output = attention(embeddings)

        self.assertEqual(output.shape, embeddings.shape)

    def test_causal_attention_hides_future_tokens(self):
        attention = MultiHeadAttention(
            num_of_heads=1,
            model_dimension=4,
            causal=True,
        )
        with torch.no_grad():
            for projection in (
                attention.query_projection,
                attention.key_projection,
                attention.value_projection,
                attention.output_projection,
            ):
                projection.weight.copy_(torch.eye(4))
                projection.bias.zero_()

        embeddings = torch.randn(1, 4, 4)
        changed_future = embeddings.clone()
        changed_future[:, 3, :] += 100.0

        original_output = attention(embeddings)
        changed_output = attention(changed_future)

        self.assertTrue(
            torch.allclose(
                original_output[:, :3, :],
                changed_output[:, :3, :],
            )
        )

    def test_invalid_head_configuration_fails(self):
        with self.assertRaises(ValueError):
            MultiHeadAttention(num_of_heads=3, model_dimension=128)

    def test_layer_norm_normalizes_each_token_vector(self):
        layer_norm = LayerNorm(model_dimension=4)
        x = torch.arange(24, dtype=torch.float32).view(2, 3, 4)

        output = layer_norm(x)

        self.assertEqual(output.shape, x.shape)
        self.assertTrue(
            torch.allclose(
                output.mean(dim=-1),
                torch.zeros(2, 3),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                output.var(dim=-1, unbiased=False),
                torch.ones(2, 3),
                atol=1e-4,
            )
        )

    def test_layer_norm_rejects_invalid_model_dimension(self):
        layer_norm = LayerNorm(model_dimension=4)

        with self.assertRaises(ValueError):
            layer_norm(torch.randn(2, 3, 5))

    def test_feed_forward_preserves_model_shape(self):
        feed_forward = FeedForward(model_dimension=128)
        embeddings = torch.randn(2, 8, 128)

        output = feed_forward(embeddings)

        self.assertEqual(output.shape, embeddings.shape)

    def test_transformer_block_preserves_model_shape(self):
        block = TransformerBlock(num_of_heads=4, model_dimension=128)
        embeddings = torch.randn(2, 8, 128)

        output = block(embeddings)

        self.assertEqual(output.shape, embeddings.shape)

    def test_transformer_block_hides_future_tokens(self):
        block = TransformerBlock(num_of_heads=2, model_dimension=4)
        embeddings = torch.randn(1, 4, 4)
        changed_future = embeddings.clone()
        changed_future[:, 3, :] += 100.0

        original_output = block(embeddings)
        changed_output = block(changed_future)

        self.assertTrue(
            torch.allclose(
                original_output[:, :3, :],
                changed_output[:, :3, :],
            )
        )

    def test_gpt_model_returns_vocabulary_logits(self):
        model = GPTModel(
            vocab_size=500,
            model_dimension=128,
            max_context_length=32,
            num_of_heads=4,
            num_of_layers=2,
        )
        token_ids = torch.randint(500, (2, 8), dtype=torch.long)

        logits = model(token_ids)

        self.assertEqual(logits.shape, (2, 8, 500))

    def test_gpt_model_hides_future_tokens(self):
        model = GPTModel(
            vocab_size=20,
            model_dimension=4,
            max_context_length=8,
            num_of_heads=2,
            num_of_layers=1,
        )
        token_ids = torch.randint(20, (1, 4), dtype=torch.long)
        changed_future = token_ids.clone()
        changed_future[:, 3] = (changed_future[:, 3] + 1) % 20

        original_logits = model(token_ids)
        changed_logits = model(changed_future)

        self.assertTrue(
            torch.allclose(
                original_logits[:, :3, :],
                changed_logits[:, :3, :],
            )
        )


if __name__ == "__main__":
    unittest.main()
