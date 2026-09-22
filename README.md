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

## Usage Examples

Install the project dependencies:
```bash
uv sync
```

**1. Pre-Training**
Train a new model and tokenizer from a text corpus:
```bash
uv run scripts/train.py \
    --text corpus.txt \
    --checkpoint checkpoints/model.pt \
    --tokenizer-path artifacts/tokenizer.json \
    --vocab-size 8192 \
    --model-dimension 512 \
    --num-layers 6 \
    --num-heads 8 \
    --context-length 512 \
    --batch-size 16 \
    --num-epochs 3 \
    --learning-rate 3e-4 \
    --device cuda
```

**2. Fine-Tuning & Resuming**
Resume training from an existing checkpoint. Use a lower learning rate for fine-tuning. (Press `Ctrl+C` to safely stop and save progress).
```bash
uv run scripts/train.py \
    --text new_corpus.txt \
    --checkpoint checkpoints/model.pt \
    --tokenizer-path artifacts/tokenizer.json \
    --vocab-size 8192 \
    --model-dimension 512 \
    --num-layers 6 \
    --num-heads 8 \
    --context-length 512 \
    --batch-size 16 \
    --num-epochs 2 \
    --learning-rate 3e-5 \
    --device cuda \
    --resume
```

**3. Text Generation**
Generate text from a trained checkpoint using a prompt:
```bash
uv run scripts/generate.py \
    --checkpoint checkpoints/model.pt \
    --prompt "The meaning of life is" \
    --max-new-tokens 150 \
    --temperature 0.8 \
    --top-k 20
```
