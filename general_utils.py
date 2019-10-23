import json
from typing import List
import numpy as np
import cv2

USEFUL_LABELS_COCO = [
    "person",
    "bicycle",
    "train",
    "car",
    "traffic light",
    "motorcycle",
    "truck",
    "bus",
    "stop sign",
]

USEFUL_LABELS_KITTI = [
    "Car",
    "Cyclist",
    "Pedestrian",
    "Van",
    "Truck",
    "Tram",
    "Person_sitting",
]

KITTI_TO_COCO_MAPPING = {
    "Car": "car",
    "Cyclist": "bicycle",
    "Pedestrian": "person",
    "Van": "car",
    "Truck": "truck",
    "Tram": "train",
    "Person_sitting": "person",
}


class TorchDataset:
    """
    Class implementing the decorator pattern to dynamically extend the behavior
    of the decorated class. In this particular case what we want to achieve is
    to wrap __getitem__ making it resize images after reading allowing torch
    to properly batch them.

    Attributes:
        data : dataset to decorate.
        resize_to : tuple indicating shape all images will be resized to.
                    Needed as PyTorch requires all the images have the same
                    shape, to batch them properly.
    """

    def __init__(self, dataset, resize_to: tuple = ()):
        self._dataset = dataset
        self.resize_to = resize_to

    # Decorated behavior implemented for __getitem__
    def __getitem__(self, index: int) -> tuple:
        sample, label = self._dataset.__getitem__(index)
        if self.resize_to:
            h, w = self.resize_to
            # opencv resize function wants a tuple (w, h) and not (h, w)
            sample = cv2.resize(sample, (w, h)) / 255.0
        return sample, label

    # Redirect all method calls to decorated class but for double underscore
    # methods which needs to be explicitly redefined
    def __getattr__(self, name):
        return getattr(self._dataset, name)

    def __repr__(self):
        return "Torch version of {}".format(str(self._dataset))

    def __len__(self):
        return len(self._dataset)


def read_json(json_file_path: str) -> dict:
    """
    This function reads json files and return it as python dict.
    """
    try:
        json_file = open(json_file_path).read()
    except FileNotFoundError:
        print(f"Oops! No such file at {json_file_path}")

    try:
        json_file = json.loads(json_file)
    except ValueError:
        print("Config file not in valid JSON format")

    return json_file


def accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
    """
    This function calculates accuracy based tp, fp, fn and tn.
    """
    if tp + tn == 0:
        return 0
    return (tp + tn) / (tp + fp + fn + tn)


def precision(tp: int, fp: int) -> float:
    """
    This function calculates precision based on
    true and false positives.
    """
    if tp == 0:
        return 0
    return tp / (tp + fp)


def recall(tp: int, fn: int) -> float:
    """
    This function calculates recall based on true positives
    and false negatives.
    """
    if tp == 0:
        return 0
    return tp / (tp + fn)


def get_average_precision(
    precision: np.array, recall: np.array, steps: int = 11
) -> float:
    """
    This function calculates the average precision.
    Definition of AP is:
        AP = 1/len(precision) * sum( p_interperp(recall) )

    And p_interp = max_(r'>r) precision(r')

    Reference:
    Everingham, Mark, et al. "The pascal visual object classes (voc)
    challenge." International journal of computer vision 88.2 (2010): 303-338

    # Parameters:
        precision: precision values for different probabilities
        recall: recall values for different probabilities
        steps: number of recall levels
    """
    assert len(precision) == len(recall), (
        "precision curve {} and recall "
        "curve {} don't have the same "
        "length".format(precision, recall)
    )

    recall_levels = np.linspace(0, 1, steps)

    rp_curve = sorted(zip(recall, precision))

    AP = 0
    for recall_level in recall_levels:
        above_thresh = [rp[1] for rp in rp_curve if rp[0] >= recall_level]
        AP += 0 if not above_thresh else max(above_thresh)

    return AP / steps


def filter_labels(dataset, useful_classes: List = []):
    """
    This function removes from datasets labels belonging to a useless class
    (not related to driving concepts).
    """
    # if useful_classes is empty return the dataset as is
    if not useful_classes:
        return dataset
    # if useful_classes contains classes not in the dataset raise an exception
    # OR we could also get rid of them from the list and continue
    for cls in useful_classes:
        if cls not in dataset.get_available_classes():
            raise Exception("{} not class of {} dataset".format(cls, dataset))

    assert isinstance(
        dataset.samples[0][-1], list
    ), "Labels should be stored in a list"
    assert isinstance(
        dataset.samples[0][-1][0], list
    ), "Each label should be in a list"
    assert isinstance(
        dataset.samples[0][-1][0][0], (str, int, float)
    ), "Each label should be a string or at least mapped to a numerical value"
    for sample in dataset.samples:
        idxs_to_delete = []
        for idx, lbl in enumerate(sample[-1]):
            cls = lbl[0]
            if cls not in useful_classes:
                idxs_to_delete.append(idx)

        # Delete elements starting from higher one, avoiding copying
        # elements in a new data structure
        for idx in sorted(idxs_to_delete, reverse=True):
            del sample[-1][idx]
    return dataset


def map_labels(dataset, mapping: dict):
    """
    This function maps labels of a dataset onto new ones given the mapping.
    """
    if not mapping:
        print("No mapping specified, dataset left unchanged")
        return dataset

    assert isinstance(
        dataset.samples[0][-1], list
    ), "Labels should be stored in a list"
    assert isinstance(
        dataset.samples[0][-1][0], list
    ), "Each label should be in a list"
    assert isinstance(
        dataset.samples[0][-1][0][0], (str, int, float)
    ), "Each labels should be a string or mapped to a numerical value"

    for sample in dataset.samples:
        for lbl in sample[-1]:
            lbl[0] = mapping.get(lbl[0], lbl[0])

    return dataset


"""
For future use of more general datasets structures here is the logic
to handle them

def _find_class_pos(dataset) -> Tuple[bool, int]:

    # This function inspects dataset to understand how it is shaped and  allow
    # other functions (filter_labels and map_labels for instance) to work  in
    # principle with any kind of dataset without strong assumption on its shape

    # [0]: Accesses the first sample of the dataset, inspecting just one is
    # enough to generalize about the dataset structure
    # [-1]: Accesses the last (second) element of the single sample. Weak
    # assumption, is reasonable to assume data are stored in a (sample, label)
    # structure. No assumptions in principle on how then label is.
    lbls_elems = dataset.samples[0][-1]

    iterable = False
    cls_idx = -1

    # object iterable and not a string (which is an iterable in principle)
    if hasattr(lbls_elems, "__iter__") and not isinstance(lbls_elems, str):
        # get (safely) shape of the iterable to then inspect a single element
        # and generalize from it
        iterable = True
        n_classes, _, _ = np.atleast_3d(np.array(lbls_elems)).shape
        # get single element of the iterable
        for idx, elem in enumerate(lbls_elems[n_classes - 1]):
            # check each element of the iterable against classes in dataset,
            # when a match is find, we have our class position
            if elem in dataset.get_available_classes():
                cls_idx = idx
                break

    # Entered the iterable, but no matching between available classes and what
    # is actually inside the iterable itself
    if iterable and cls_idx == -1:
        raise Exception(
            "Can't find class position in the iterable, please check that "
            "classes match classes returned by get_available_classes"
        )

    return iterable, cls_idx
"""
