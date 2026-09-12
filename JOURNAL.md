# Project Journey & Optimizations

## Custom Rust BPE Engine
- Built a native Rust BPE tokenizer from scratch using PyO3 and Maturin to interface with Python.
- Optimized the Rust training loop with `Rayon` and `FxHashMap` for 12-thread parallel chunk processing, drastically slashing tokenization time on massive 500MB datasets.
- Created a Python `LLMTokenizer` wrapper class to seamlessly bridge the Rust backend with custom PyTorch special tokens (e.g. `<|user|>`, `<|bot|>`, `<|endoftext|>`) while maintaining a perfect Tensor Core-aligned vocabulary size (exactly 8192).

## Data Preparation & Fine-Tuning
- **Pre-Training (WikiText):** Processed HuggingFace Wikipedia Parquet files into a unified 500MB `wikitext.txt` corpus. Dynamically injected `<|endoftext|>` markers at document boundaries (using regex headers) to preserve context cleanly and prevent cross-article contamination.
- **Instruction Tuning (Philosophy):** Developed a parsing script (`create_chat_dataset.py`) to convert raw Project Gutenberg philosophy texts into a 33,000+ turn Chatbot dataset formatted precisely with `<|user|>` and `<|bot|>` tokens.

## Model Pipeline & Architecture
- Constructed a GPT architecture entirely from scratch using PyTorch (embeddings, multi-head causal attention, feed-forward MLPs).
- Resolved tricky device-mapping bugs (ensuring the model is assigned to the GPU `cuda` device *before* passing its parameters to the AdamW optimizer to prevent CPU/GPU momentum buffer crashes).
- Added graceful `KeyboardInterrupt` exception handling to safely catch manual stops (`Ctrl+C`) and preserve the `.pt` checkpoint instantly.
- Built a live `scripts/chat.py` interface to interact with the trained LLM, featuring a dynamic auto-stopping mechanism and continuous conversation history tracking perfectly synced with the custom Tokenizer wrapper.

## Future Improvements to Explore
- **KV Caching for Generation:** Update the `GPTModel` and `generate_tokens` to cache the Key and Value matrices during inference. Currently, the chat script recomputes the entire context mathematically for every single new word. KV Caching makes Chatbot response generation instantly fast ($O(N)$ instead of $O(N^2)$).
- **Reasoning & Chain-of-Thought:** Introduce special `<|thought|>` tags into the Chatbot dataset to train the model to "think out loud" before answering. This allows the model to perform multi-step logical deduction and self-correction, mimicking the reasoning capabilities of advanced models like OpenAI's o1 or DeepSeek-R1.
