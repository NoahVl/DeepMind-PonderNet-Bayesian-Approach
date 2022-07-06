import os
from itertools import cycle
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ParityDataset(Dataset):
    """
    PyTorch Datset for the parity problem as introduced in the original ACT paper (Graves, 2016).
    """

    def __init__(self, filepath: str):
        self.problems, self.labels = torch.load(filepath)

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx: int):
        return self.problems[idx], self.labels[idx]


def generate_parity_data(
    vector_size: int,
    num_problems: int,
    min_integer_change: int = None,
    max_integer_change: int = None,
    uniform: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a dataset of parity problems.
    :param vector_size: The size of the vectors to be generated.
    :param num_problems: The number of problems to be generated.
    :param min_integer_change: The minimum number of integers that should be changed to a -1 or 1.
    :param max_integer_change: The maximum number of integers that should be changed to a -1 or 1.
    :return: A tuple containing the problems of size (num_problems, vector_size) and the labels of size (num_problems,).
    """

    # If no min/max integer change is specified, assume we want the non extrapolation case.
    if min_integer_change is None or max_integer_change is None:
        min_integer_change = 1
        max_integer_change = vector_size

    problems = torch.zeros((num_problems, vector_size))
    labels = torch.zeros(num_problems, dtype=torch.int64)

    num_integer_change = cycle(range(min_integer_change, max_integer_change + 1))
    for index, problem in enumerate(problems):
        # Sample which indices should be changed from 0 to -1 or 1.
        num_changes = (
            next(num_integer_change)
            if uniform
            else np.random.randint(min_integer_change, max_integer_change + 1)
        )
        change_indices = np.random.choice(
            np.arange(vector_size),
            size=num_changes,
            replace=False,
        )
        # Change the indices to -1 or 1.
        problem[change_indices] = torch.Tensor(
            np.random.choice([-1, 1], len(change_indices), replace=True)
        )

        # The label is 0 if the number of +/-1s is even, otherwise it is 1.
        labels[index] = problem.sum() % 2

    return problems, labels


def save_parity_data(
    vector_size: int,
    num_problems: tuple[int, int, int],
    path: str,
    extrapolate: bool = False,
    uniform: bool = False,
):
    """
    Saves the parity problems and labels to a file; for the train, valid and test dataset respectively.
    :param vector_size: The size of the vectors to be generated.
    :param num_problems: The number of problems to be generated.
    :param path: The path to the file to save the data to.
    :param extrapolate: Whether to generate the extrapolation case or not.
    """
    if extrapolate:
        train_data = generate_parity_data(
            vector_size, num_problems[0], 1, vector_size // 2, uniform=uniform
        )
        valid_data = generate_parity_data(
            vector_size, num_problems[1], 1, vector_size // 2, uniform=uniform
        )
        test_data = generate_parity_data(
            vector_size,
            num_problems[2],
            vector_size // 2 + 1,
            vector_size,
            uniform=uniform,
        )
    else:
        train_data = generate_parity_data(vector_size, num_problems[0], uniform=uniform)
        valid_data = generate_parity_data(vector_size, num_problems[1], uniform=uniform)
        test_data = generate_parity_data(vector_size, num_problems[2], uniform=uniform)

    problem_str = f"{vector_size}{'_extrapolate' if extrapolate else ''}"

    os.makedirs(path, exist_ok=True)
    torch.save(train_data, os.path.join(path, f"train_{problem_str}.pt"))
    torch.save(valid_data, os.path.join(path, f"valid_{problem_str}.pt"))
    torch.save(test_data, os.path.join(path, f"test_{problem_str}.pt"))


def create_parity_dataloaders(
    path: str, batch_size: int, num_workers: int, vector_size=48, extrapolate=False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates the dataloaders for the parity problems.
    :param path: The path to the parity problems.
    :param batch_size: The batch size.
    :param num_workers: The number of workers to use.
    :param vector_size: The size of the vectors to be generated.
    :param extrapolate: Whether to generate the extrapolation case or not.
    :return: A tuple containing the dataloaders for the training, validation and test sets.
    """
    problem_str = f"{vector_size}{'_extrapolate' if extrapolate else ''}"

    train_dataset = ParityDataset(os.path.join(path, f"train_{problem_str}.pt"))
    valid_dataset = ParityDataset(os.path.join(path, f"valid_{problem_str}.pt"))
    test_dataset = ParityDataset(os.path.join(path, f"test_{problem_str}.pt"))

    return (
        DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
        DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers),
        DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers),
    )


if __name__ == "__main__":
    size_vector = 10
    num_probs = (100000, 10000, 10000)
    save_dir = "./"
    extrap = False

    save_parity_data(
        vector_size=size_vector,
        num_problems=num_probs,
        path=save_dir,
        extrapolate=extrap,
    )
    train_loader, valid_loader, test_loader = create_parity_dataloaders(
        save_dir,
        vector_size=size_vector,
        batch_size=32,
        num_workers=1,
        extrapolate=extrap,
    )

    # Print the first batch of the training set
    print("Training set:")
    for _, (problems, labels) in enumerate(train_loader):
        for problem, label in zip(problems, labels):
            print(problem, label)

        break

    # Print the first batch of the validation set (to test if extrapolation is working)
    print("\n\nValidation set:")
    for _, (problems, labels) in enumerate(valid_loader):
        for problem, label in zip(problems, labels):
            print(problem, label)

        break
