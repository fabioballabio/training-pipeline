import numpy as np
from importlib import import_module
from datetime import timedelta
from typing import List
from collections import Counter
import torch
import torchvision
import time
import re
from datasets.datasets import BasicDataset
from general_utils import filter_labels


def get_model(model_name: str):
    """
    This function returns the model class given a string indicating
    the desired model
    """
    # Assumptio is that following PEP8 specs, MyNetModel should reside in
    # my_net_model.py, so spit model name on capital letters, reorganize them
    # with underscores getting file name out of model name
    model_file_blocks = re.findall("[A-Z][^A-Z]*", model_name)
    model_file_blocks = [block.lower() for block in model_file_blocks]
    model_file = "_".join(model_file_blocks)
    model_path = "models." + model_file
    # Try to import model from our custom models defined in ./models package
    try:
        model_module = import_module(model_path)
        model_class = getattr(model_module, model_name)
        print("{} model found in models.".format(model_name))
    # if not found there look for PyTorch native implementations
    except (ModuleNotFoundError, AttributeError):
        print(
            "Model {} not found in models.{}.py\n"
            "Looking for it in PyTorch predefined models...".format(
                model_name, model_file
            )
        )
        try:
            model_class = getattr(torchvision.models, model_name)
            print("{} model found in PyTorch.".format(model_name))
        # if not found neither in PyTorch exit program with an error msg
        except AttributeError:
            # sleep gives the program time to flush printing buffers before
            # the exit msg
            time.sleep(0.5)
            exit("{} model not found, program stopped".format(model_name))

    model = model_class()
    return model


def build_classification_dataset(dataset: BasicDataset, keep_class: List = []):
    """
    This function reshapes a dataset meant for detection to a  dataset meant
    for classification
    """
    if not keep_class:
        print(
            "No class to keep specified, classification dataset not built. "
            "Returned the same dataset feed"
        )
        return dataset
    dataset = filter_labels(dataset, useful_classes=keep_class)
    for idx in range(len(dataset.samples)):
        dataset.samples[idx] = list(dataset.samples[idx])
        if not dataset.samples[idx][-1]:
            # List is empty so no cars in the image
            dataset.samples[idx][-1] = 0.0
        else:
            dataset.samples[idx][-1] = 1.0

        dataset.samples[idx] = tuple(dataset.samples[idx])

    return dataset


def rebalance_dataset(dataset: BasicDataset) -> BasicDataset:
    """
    Makes an umbalanced classification dataset balanced
    """
    pos_class = 0
    neg_class = 0
    tot = 0
    spare = 0
    for sample in dataset.samples:
        tot += 1
        _, label = sample
        if int(label) == 0:
            neg_class += 1
        elif int(label) == 1:
            pos_class += 1
        else:
            spare += 1

    if spare != 0:
        raise Exception(
            "Ill posed binary classification problem, "
            "some data are neither postive class, nor negative class "
        )

    if pos_class == neg_class:
        return dataset

    elif pos_class > neg_class:
        unbalance_factor = pos_class / neg_class
        n_to_del = round(
            len(dataset.samples) * (1 - 2 / (1 + unbalance_factor))
        )
        idxs_to_del = []
        for idx in range(len(dataset.samples)):
            if dataset.samples[idx][-1] == 1:
                idxs_to_del.append(idx)
    else:
        unbalance_factor = neg_class / pos_class
        n_to_del = round(
            len(dataset.samples) * (1 - 2 / (1 + unbalance_factor))
        )
        idxs_to_del = []
        for idx in range(len(dataset.samples)):
            if dataset.samples[idx][-1] == 0:
                idxs_to_del.append(idx)

    # delete in reverse order to avoid problems with indeces
    idxs_to_del = idxs_to_del[-n_to_del:]
    for idx in sorted(idxs_to_del, reverse=True):
        del dataset.samples[idx]
    return dataset


def get_labels_counts(dataset: BasicDataset):
    """
    This function returns some trivial stats about a classification dataset
    """
    labels = [sample[-1] for sample in dataset.samples]
    labels_dict = Counter(labels)
    tot = sum(labels_dict.values())
    print("Dataset represents labels as follows:")
    for lbl, freq in labels_dict.items():
        print(
            "{} ({:.2f}%) data points belong to class {}".format(
                freq, freq / tot, lbl
            )
        )
    return labels_dict


"""
def get_accuracy(
        preds: List,
        ground_truth: List,
        acceptance_thresh: float = 0.5) -> float:
    '''
    This functions returns the accuracy given classification predictions
    and ground truth
    '''
    preds = np.array(preds).flatten()
    ground_truth = np.array(ground_truth).flatten()
    n_preds = len(ground_truth)
    preds_binary = np.where(preds >= acceptance_thresh, 1, 0)
    correct_preds = np.sum(preds_binary == ground_truth)
    return correct_preds / n_preds
"""


def get_metrics(
    preds: List, ground_truth: List, acceptance_thresh: float = 0.5
) -> tuple:
    preds = np.array(preds).flatten()
    ground_truth = np.array(ground_truth).flatten()
    preds_binary = np.where(preds >= acceptance_thresh, 1, 0)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for pred, gt in zip(preds_binary, ground_truth):
        if pred == 1 and gt == 1:
            tp += 1
        elif pred == 0 and gt == 0:
            tn += 1
        elif pred == 1 and gt == 0:
            fp += 1
        else:
            fn += 1
    return tp, fp, fn, tn


def format_time(elapsed_seconds: float) -> str:
    """
    This function calculates days, hours, minutes and seconds out of
    elapsed seconds and returns a well formatted string containing these infos
    """
    elapsed_seconds = timedelta(seconds=elapsed_seconds)
    days, hours, mins, secs = (
        elapsed_seconds.days,
        (elapsed_seconds.seconds // 3600) % 24,
        (elapsed_seconds.seconds // 60) % 60,
        elapsed_seconds.seconds % 60,
    )
    if days != 0:
        return "{} days, {} hours, {} minutes and {} seconds".format(
            days, hours, mins, secs
        )
    elif hours != 0:
        return "{} hours, {} minutes and {} seconds".format(hours, mins, secs)
    elif mins != 0:
        return "{} minutes and {} seconds".format(mins, secs)
    else:
        return "{} seconds".format(secs)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
