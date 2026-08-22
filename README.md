# LLM from Scratch

A small, learning-oriented GPT implementation built with PyTorch and a Rust BPE
tokenizer.

## Layout

- `src/model/`: embeddings, attention, Transformer blocks, GPT.
- `src/training/`: loss, evaluation, optimization loop.
- `scripts/train.py`: train from a local plain-text UTF-8 corpus.
- `scripts/generate.py`: sample from a saved checkpoint.
- `tests/`: shape, causality, loss, and generation checks.

## Train

Pass a local plain-text corpus. The script creates a 90/10 train/validation
split, trains the BPE tokenizer only on the training split, and saves the
tokenizer and final checkpoint.
