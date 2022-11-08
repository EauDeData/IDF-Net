import torch
from typing import *

# https://open.spotify.com/track/4hg4T0JwkQpBFRJJg4U3x7?si=bfbc46919ad14798

class IDFNetDataLoader(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super(IDFNetDataLoader).__init__()

    def __len__(self) -> int:
        raise NotImplementedError("Make sure you implement an iterable of text in your dataset.")
    
    def call(self) -> Tuple[torch.tensor]:
        raise NotImplementedError("Make sure you implement an iterable of text in your dataset.")

    def iter_text(self) -> Iterable:
        raise NotImplementedError("Make sure you implement an iterable of text in your dataset.")