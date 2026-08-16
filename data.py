from torch.utils.data import Dataset, DataLoader
from tokenizer import SimpleTokenizer
import torch
import config

class GPTDataset(Dataset):
    def __init__(self, txt: str, tokenizer: SimpleTokenizer, max_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []

        # tokenize the entire text
        token_ids = tokenizer.encode(txt)
        assert len(token_ids) > max_length

        # use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
        text,
        tokenizer: SimpleTokenizer,
        batch_size=config.BATCH_SIZE,
        max_length=config.MAX_LENGTH,
        stride=config.STRIDE,
        shuffle=True,
        drop_last=True,
        num_workers=0
) -> DataLoader:
    """
    Creates a DataLoader for the GPTDataset.
    
    Args:
        text: The text to be tokenized and used for the dataset
        tokenizer: The tokenizer to use for encoding the text
        batch_size: The number of samples per batch to load
        max_length: The maximum length of each sequence
        stride: The stride size for the sliding window
        shuffle: Whether to shuffle the dataset at every epoch
        drop_last: Whether to drop the last incomplete batch
        num_workers: How many subprocesses to use for data loading
    """
    dataset = GPTDataset(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

def next_batch(loader: DataLoader):
    embedding_layer = torch.nn.Embedding(config.VOCAB_SIZE, config.EMBEDDING_DIM)

    return embedding_layer(next(iter(loader))[0]), embedding_layer(next(iter(loader))[1])




