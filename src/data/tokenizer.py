
from pathlib import Path

from bpe_tokenizer import Tokenizer as RustTokenizer


def train_tokenizer(
    texts: list[str],
    target_vocab_size: int,
) -> RustTokenizer:
    """Returns default tokenizer if texts is empty, or target_vocab_size below 257"""
    
    tokenizer = RustTokenizer.train(texts, target_vocab_size)
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


def train_or_load_tokenizer(
    texts: list[str],
    path: str | Path,
    target_vocab_size: int,
) -> RustTokenizer:
    source = Path(path)

    if source.exists() and source.is_dir():
        raise IsADirectoryError(f"Tokenizer path is a directory: {source}")

    if not source.exists():
        tokenizer = train_tokenizer(texts, target_vocab_size)
        save_tokenizer(tokenizer, source)
        return tokenizer

    tokenizer = load_tokenizer(source)

    if tokenizer.vocab_size() < target_vocab_size:
        tokenizer = train_tokenizer(texts, target_vocab_size)
        save_tokenizer(tokenizer, source)

    return tokenizer
