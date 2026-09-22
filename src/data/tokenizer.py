import re
from pathlib import Path

from bpe_tokenizer import Tokenizer as RustTokenizer


def train_tokenizer(
    text: str,
    target_vocab_size: int,
) -> RustTokenizer:
    """Returns default tokenizer if text is empty, or target_vocab_size below 257"""
    
    tokenizer = RustTokenizer.train(text, target_vocab_size)
    return tokenizer
    
def save_tokenizer(
    tokenizer: RustTokenizer,
    path: str | Path,
) -> None:
    destination = Path(path)

    if destination.exists() and destination.is_dir():
        raise IsADirectoryError(f"Tokenizer path is a directory: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(destination))
    
def load_tokenizer(path: str | Path) -> RustTokenizer:
    source = Path(path)

    if not source.exists():
        raise FileNotFoundError(f"Tokenizer file does not exist: {source}")
    if not source.is_file():
        raise IsADirectoryError(f"Tokenizer path is not a file: {source}")

    return RustTokenizer.load(str(source))


class LLMTokenizer:
    """
    A Python wrapper that manages special tokens
    """
    def __init__(self, base_tokenizer: RustTokenizer):
        self.base = base_tokenizer
        self.base_vocab_size = base_tokenizer.vocab_size()
    
        self.special_tokens = {
            "<|endoftext|>": self.base_vocab_size,     # 8189
            "<|user|>": self.base_vocab_size + 1,      # 8190
            "<|bot|>": self.base_vocab_size + 2        # 8191
        }
        self.inv_special_tokens = {v: k for k, v in self.special_tokens.items()}
        self.special_token_pattern = re.compile(
            "(" + "|".join(re.escape(token) for token in self.special_tokens) + ")"
        )
        
    def vocab_size(self):
        return self.base_vocab_size + len(self.special_tokens)
        
    def encode(self, text: str) -> list[int]:
        ids: list[int] = []

        for part in self.special_token_pattern.split(text):
            if not part:
                continue

            special_token_id = self.special_tokens.get(part)
            if special_token_id is not None:
                ids.append(special_token_id)
            else:
                ids.extend(self.base.encode(part))

        return ids

    def decode(self, ids: list[int]) -> str:
        result = []
        current_chunk = []
        
        for idx in ids:
            if idx in self.inv_special_tokens:
                if current_chunk:
                    result.append(self.base.decode(current_chunk))
                    current_chunk = []
                result.append(self.inv_special_tokens[idx])
            else:
                current_chunk.append(idx)
                
        if current_chunk:
            result.append(self.base.decode(current_chunk))
            
        return "".join(result)

def train_or_load_tokenizer(
    text: str,
    path: str | Path,
    target_vocab_size: int,
) -> LLMTokenizer:
    source = Path(path)

    if source.exists() and source.is_dir():
        raise IsADirectoryError(f"Tokenizer path is a directory: {source}")

    if not source.exists():
        tokenizer = train_tokenizer(text, target_vocab_size - 3)
        save_tokenizer(tokenizer, source)
        return LLMTokenizer(tokenizer)

    tokenizer = load_tokenizer(source)

    if tokenizer.vocab_size() < target_vocab_size - 3:
        tokenizer = train_tokenizer(text, target_vocab_size - 3)
        save_tokenizer(tokenizer, source)

    return LLMTokenizer(tokenizer)
