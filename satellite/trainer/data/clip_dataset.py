from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import numpy as np
import pandas as pd

from .utils import fast_scandir

def coord2img(coord):
    # use google earth engine API to download rgb sat image given a coordinate point
    # sort by cloudiness and choose less cloudy one

    return "some rgb img"

class CLIPDataset(Dataset):
    def __init__(
        self,
        data_path,
        processor,
    ):

        self.processor = processor
        
        df = pd.read_csv(data_path)

        self.coordinates = []
        texts = [] # sks for nuraghe yes and ksk for nuraghe no
        
          
        print("Pre tokenizing dataset...")
        self.tokens = self.processor(text=self.texts, return_tensors='pt', padding=True)

        print(f"Tokenized Dataset shape: {self.tokens['input_ids'].shape}")

        assert self.tokens['input_ids'].shape[0] == len(self.coordinates), f"Tokenized dataset length {self.tokens['input_ids'].shape[0]} does not match coordinates length {len(self.coordinates)}"
            
    def __getitem__(
        self, idx: Union[Tensor, int]
    ):

        idx = idx.tolist() if torch.is_tensor(idx) else idx  # type: ignore

        img = coord2img(self.coordinates[idx])

        # process image
        pixel_values = self.processor(images=img)['pixel_values']

                 
        output = dict(
            input_ids=self.tokens['input_ids'][idx],
            attention_mask=self.tokens['attention_mask'][idx],
            pixel_values=pixel_values,
        )
        
        return output

    def __len__(self) -> int:
        return len(self.coordinates)
