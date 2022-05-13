import os
from typing import List

from torchvision import datasets


class SelectedImageFolder(datasets.ImageFolder):
    selected_classes: List[str] = []
    """
    We use this class in order to pick parts of the dataset and move from
    Covid vs Normal vs Virus
    to 
    Covid vs Virus and so on
    """

    def __init__(self, selected_classes: List[str], *args, **kwargs):
        self.selected_classes = selected_classes
        super(SelectedImageFolder, self).__init__(*args, **kwargs)
        print(*args)

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir),
            and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        # Takes all classes whitin the directory dir
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        # Check if the passed selected_classes belongs to all_classes
        is_subset = set(self.selected_classes).issubset(set(classes))
        if is_subset and self.selected_classes:
            # Set the classes with the custom list
            classes = self.selected_classes
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
