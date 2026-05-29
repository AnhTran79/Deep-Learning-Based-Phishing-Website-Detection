from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from sklearn.model_selection import train_test_split


T = TypeVar("T")


def split_train_val_test(
    records: Sequence[T],
    labels: Sequence[int],
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[list[T], list[T], list[T]]:
    record_list = list(records)
    label_list = list(labels)
    if len(record_list) != len(label_list):
        raise ValueError("records and labels must have the same length.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0.0 <= val_size < 1.0:
        raise ValueError("val_size must be between 0 and 1.")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be less than 1.")

    indices = list(range(len(record_list)))
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        stratify=label_list,
        random_state=random_state,
    )
    if val_size == 0:
        return [record_list[index] for index in train_val_indices], [], [record_list[index] for index in test_indices]

    train_val_labels = [label_list[index] for index in train_val_indices]
    relative_val = val_size / (1.0 - test_size)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=relative_val,
        stratify=train_val_labels,
        random_state=random_state,
    )
    return (
        [record_list[index] for index in train_indices],
        [record_list[index] for index in val_indices],
        [record_list[index] for index in test_indices],
    )
