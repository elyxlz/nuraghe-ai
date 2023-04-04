from typing import List, Tuple, TypeVar, Sequence
import os
import math
from torch.utils.data import Dataset, Subset
import torch
T = TypeVar("T")



def fractional_random_split(
    dataset: Dataset[T], fractions: Sequence[int]
) -> List[Subset[T]]:
    """Fractional split that follows the same convention as random_split"""
    assert sum(fractions) == 1.0, "Fractions must sum to 1.0"

    length = len(dataset)  # type: ignore[arg-type]
    indices = torch.randperm(length)
    splits = []
    cursor = 0

    for fraction in fractions:
        next_cursor = math.ceil(length * fraction + cursor)
        splits += [Subset(dataset, indices[cursor:next_cursor])]  # type: ignore[arg-type] # noqa
        cursor = next_cursor

    return splits