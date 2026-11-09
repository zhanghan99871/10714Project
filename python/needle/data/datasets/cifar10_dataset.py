import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        super().__init__(transforms=transforms)

        self.p = p

        if train:
            filenames = [f"data_batch_{i}" for i in range(1, 6)]
        else:
            filenames = ["test_batch"]

        images = []
        labels = []

        for filename in filenames:
            path = os.path.join(base_folder, filename)

            with open(path, "rb") as f:
                batch = pickle.load(f, encoding="bytes")

            images.append(batch[b"data"])
            labels.extend(batch[b"labels"])

        X = np.concatenate(images, axis=0)

        # (N, 3072) -> (N, 3, 32, 32)
        X = X.reshape(-1, 3, 32, 32)

        # Normalize to [0, 1]
        self.X = X.astype(np.float32) / 255.0
        self.y = np.array(labels, dtype=np.int64)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        image, label = self.X[index], self.y[index]
        if self.transforms:
            image = image.transpose(1, 2, 0)
            image = self.apply_transforms(image)
            image = image.transpose(2, 0, 1)

        return image, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return len(self.y) 
        ### END YOUR SOLUTION
