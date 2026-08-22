import torch
from bpe_tokenizer import Tokenizer
from torch.utils.data import DataLoader, Dataset


class SimpleDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        text: str,
        tokenizer: Tokenizer,
        max_length: int,
        stride: int,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be greater than zero")
        if stride <= 0:
            raise ValueError("stride must be greater than zero")
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        token_ids = tokenizer.encode(text)
        required_tokens = max_length + 1
        if len(token_ids) < required_tokens:
            raise ValueError(
                "text does not contain enough tokens for one complete "
                f"input/target pair: need {required_tokens}, got {len(token_ids)}"
            )

        self.token_ids = token_ids
        self.max_length = max_length
        self.stride = stride

    def __len__(self) -> int:
        # A target sequence is shifted by one position, so every window needs
        # max_length + 1 source tokens.
        return 1 + (len(self.token_ids) - self.max_length - 1) // self.stride

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(f"dataset index out of range: {index}")

        start = index * self.stride
        input_ids = self.token_ids[start : start + self.max_length]
        target_ids = self.token_ids[start + 1 : start + self.max_length + 1]

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )


def createDataLoader(
    txt: str,
    tokenizer: Tokenizer,
    batch_size: int,
    max_length: int,
    stride: int,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    dataset = SimpleDataset(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
