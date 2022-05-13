import os
from typing import List, Dict
from random import shuffle, seed
from shutil import copy2

from src.common.utils import get_env, load_envs

load_envs()
BASE_PATH: str = get_env("BASE_PATH")


def filter_right_extension(file_name: str) -> bool:
    allowed_extension = ["jpeg", "png", "jpg"]
    return file_name.split(".")[-1] in allowed_extension


def calculate_percentage(full_size, percentage) -> int:
    return (full_size // 100) * percentage


def split_files(file_path: str, train_percentage: int, dev_percentage: int) -> Dict[str, List[str]]:
    img_list = os.listdir(file_path)
    filtered_images = list(filter(filter_right_extension, img_list))
    shuffle(filtered_images)
    test_percentage = 100 - train_percentage - dev_percentage

    train_size, dev_size, test_size = (
        calculate_percentage(len(filtered_images), train_percentage),
        calculate_percentage(len(filtered_images), dev_percentage),
        calculate_percentage(len(filtered_images), test_percentage),
    )
    print("Dataset stats: Train size {}, Dev size {} Test size {}".format(train_size, dev_size, test_size))
    result: Dict[str, List[str]] = {
        "train": filtered_images[: train_size + 1],
        "validation": filtered_images[train_size + 1 : train_size + dev_size + 1],
        "test": filtered_images[train_size + dev_size + 1 :],
    }
    return result


DEBUG = False
if DEBUG:
    dataset_path: str = f"{BASE_PATH}/data/branch-1/dataset/train/"
    class_list = os.listdir(dataset_path)
    train, dev, test = split_files(dataset_path + class_list[0] + "/", 70, 20)

if __name__ == "__main__" and not DEBUG:
    dest_path: str = f"{BASE_PATH}/data/branch-2/dataset/"
    os.makedirs(dest_path, exist_ok=True)
    seed(1234)
    dataset_path: str = f"{BASE_PATH}/data/branch-1/dataset/train/"
    class_list = os.listdir(dataset_path)
    for label in class_list:
        splitted: Dict[str, List[str]] = split_files(dataset_path + label + "/", 70, 20)
        for dataset_type, img_list in splitted.items():
            os.makedirs(dest_path + dataset_type + "/" + label + "/", exist_ok=True)
            for img in img_list:
                copy2(dataset_path + label + "/" + img, dest_path + dataset_type + "/" + label + "/" + img)
