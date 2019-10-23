import os
import cv2
import pandas as pd
from general_utils import read_json


class BasicDataset:
    """
    Basic dataset structure.

    # Parameters:
        root_dir (string): Directory with all the data.
    """

    def __init__(self, root_dir: str = "./data"):
        self.root_dir = root_dir
        self.samples = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple:
        """
        # Parameters:
            index (int): Index
        # Returns:
            tuple: (sample, target) where target is class_index of class
        """
        path, target = self.samples[index]
        sample = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        return sample, target

    def _makedataset(self):
        raise NotImplementedError(
            "Subclasses should implement _makedataset method!"
        )

    def get_available_classes(self):
        raise NotImplementedError(
            "Subclasses should implement get_available_classes method!"
        )


class KittiDataset(BasicDataset):
    """
    Kitti dataset structure.

    # Parameters:
        root_dir (string): Directory with all the data.
    """

    def __init__(self, root_dir: str = "./data"):
        super(KittiDataset, self).__init__(root_dir)
        self.img_dir = os.path.join(self.root_dir, "training", "image_2")
        self.lbl_dir = os.path.join(self.root_dir, "training", "label_2")
        self.samples = self._makedataset()

    def __repr__(self) -> str:
        return "Kitti"

    def _makedataset(self) -> dict:
        """
        Function to build the dataset from raw

        # Returns:
            images dict: data stored as key, value pair in which key is
                        sample_path, and values are bboxes and classes.
        """
        images = []
        # Get img, class tuples
        keys = sorted(
            [
                fname.split(".")[0]
                for fname in os.listdir(self.img_dir)
                if fname.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        img_fname = sorted(
            [
                fname
                for fname in os.listdir(self.img_dir)
                if fname.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        lbl_fname = sorted(
            [
                fname
                for fname in os.listdir(self.lbl_dir)
                if fname.lower().endswith(".txt")
            ]
        )
        img_dict = dict(zip(keys, img_fname))
        lbl_dict = dict(zip(keys, lbl_fname))
        """
        Filtering based on following assumptions:
            - 0: class idx
            - 4: x0 bbox idx
            - 5: y0 bbox idx
            - 6: x1 bbox idx
            - 7: y1 bbox idx
        """
        for key in img_dict.keys():
            img_path = os.path.join(self.img_dir, img_dict[key])
            lbl_path = os.path.join(self.lbl_dir, lbl_dict[key])
            with open(lbl_path, "r") as txt_file:
                sample_cls = []
                for line in txt_file:
                    field_list = line.split()
                    line_cls = field_list[0]
                    x0, y0, x1, y1 = [
                        float(field) for field in field_list[4:8]
                    ]
                    bbox = (x0, y0, x1 - x0, y1 - y0)
                    sample_cls.append([line_cls, 1, bbox])
            item = (img_path, sample_cls)
            images.append(item)

        return images

    def get_available_classes(self) -> set:
        """
        This function returns set of classes present in the dataset.
        """
        classes = set()
        for image in self.samples:
            for cls in image[-1]:
                classes.add(cls[0])
        return classes


class COCODataset(BasicDataset):
    """
    COCO dataset structure.

    # Parameters:
        root_dir (string): Directory with all the data.
    """

    def __init__(self, root_dir: str = "./data"):
        super(COCODataset, self).__init__(root_dir)
        self.data_dir = os.path.join(
            self.root_dir, "annotations", "instances_val2017.json"
        )
        self.raw_data_dir = os.path.join(self.root_dir, "val2017")
        self.samples = self._makedataset()

    def __repr__(self) -> str:
        return "COCO"

    def _makedataset(self) -> dict:
        """
        Function to build the dataset from raw .json file

        # Parameters:
            images dict: data stored as key, value pair in which key is
                        sample_path, and values are bboxes and classes.
        """
        json_COCO = read_json(self.data_dir)
        images_df = pd.DataFrame(json_COCO["images"])
        annotations_df = pd.DataFrame(json_COCO["annotations"])
        categories_df = pd.DataFrame(json_COCO["categories"])

        merged_df = images_df.merge(
            annotations_df, how="inner", left_on="id", right_on="image_id"
        )
        merged_df = merged_df.merge(
            categories_df, how="inner", left_on="category_id", right_on="id"
        )
        df_by_filename = merged_df.groupby("file_name", as_index=False)

        images = []
        for fname, group in df_by_filename:
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(self.raw_data_dir, fname)
                sample_cls = [
                    [name, 1, bbox]
                    for name, bbox in zip(group["name"], group["bbox"])
                ]
                item = (img_path, sample_cls)
                images.append(item)

        return images

    def get_available_classes(self) -> set:
        """
        This function returns set of classes present in the dataset.
        """
        classes = set()
        for image in self.samples:
            for cls in image[-1]:
                classes.add(cls[0])
        return classes
