import torch
from torchvision.datasets.imagenet import ImageNet, load_meta_file
from torchvision.datasets.folder import find_classes, has_file_allowed_extension
from torchvision.datasets.utils import verify_str_arg
from typing import Any, Dict, Union, List, Optional, Tuple, Callable, cast
import os
import json

def json_to_dict(split: str = "base") -> Dict[str, Any]:
    json_path = os.path.join(os.path.dirname(__file__), "miniimagenet", f"{split}.json")
    with open(json_path, "r") as f:
        file_list = json.load(f)

        wnids = file_list["label_names"]
        image_names = file_list["image_names"]
        idx = file_list["image_labels"]
        return wnids, image_names, idx


def make_dataset(
    directory: str,
    split: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

    # TODO check if it's necessary to change class_to_idx
    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)

    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")


    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()

    wnids, image_names, idx = json_to_dict(split)

    wnids = set()

    for image_name in image_names:
        path = os.path.join(directory, image_name)
        if is_valid_file(path):
            wnid = image_name.split("/")[0]
            wnids.add(wnid)
            idx = class_to_idx[wnid]
            item = path, idx
            instances.append(item)

    available_classes = wnids

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances


class MiniImageNet(ImageNet):

    def __init__(self, root: str, split: str = "base", **kwargs: Any) -> None:
        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("base", "val", "novel"))

        self.parse_archives()
        wnid_to_classes = load_meta_file(self.root)[0]
        super(ImageNet, self).__init__(self.split_folder, **kwargs)
        self.root = root

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx for idx, clss in enumerate(self.classes) for cls in clss}

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, "train")


    def make_dataset(
        self,
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")


        return make_dataset(directory, self.split, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders of miniimagenet with json files
        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        wnids, _, idx = json_to_dict(self.split)
        class_to_idx = {}
        for i, wnid in enumerate(wnids):
            if i in idx:
                class_to_idx[wnid] = i

        classes = list(class_to_idx.keys())
        return classes, class_to_idx

    def get_num_elem_per_class(self):
        idx = set()
        for sample in self.samples:
            idx.add(sample[1])

        # create a dict with keys idx and values the number of elements per class
        num_elem_per_class = {}
        for i in idx:
            num_elem_per_class[i] = 0
        for sample in self.samples:
            num_elem_per_class[sample[1]] += 1

        # verify that the number of elements per class is the same for all classes
        assert len(set(num_elem_per_class.values())) == 1

        return set(num_elem_per_class.values()).pop()