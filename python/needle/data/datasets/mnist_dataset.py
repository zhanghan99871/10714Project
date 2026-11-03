from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        import gzip
        import struct 
        image_list = []
        with gzip.open(image_filename, "rb") as f:
            _ = struct.unpack(">I", f.read(4))
            num_images = struct.unpack(">I", f.read(4))[0]
            num_rows = struct.unpack(">I", f.read(4))[0]
            num_columns = struct.unpack(">I", f.read(4))[0]
            for i in range(num_images):
                image = struct.unpack(">{}B".format(num_rows*num_columns), f.read(num_rows*num_columns))
                image_np = np.array(image, dtype=np.float32).reshape((num_columns, num_rows, 1)) / 255.0 
                image_list.append(image_np) 
        image_list = np.array(image_list)
        label_list = []
        with gzip.open(label_filename, "rb") as f:
            _ = struct.unpack(">I", f.read(4))
            num_labels = struct.unpack(">I", f.read(4))[0]
            for i in range(num_labels):
                label = struct.unpack(">B", f.read(1))[0]
                label_list.append(label) 
        label_list = np.array(label_list, dtype=np.uint8) 
        self.image_list = image_list 
        self.label_list = label_list 
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        x = self.image_list[index]
        if self.transforms is not None:
            for each in self.transforms:
                x = each(x) 
        return x, self.label_list[index] 
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return len(self.label_list)
        ### END YOUR SOLUTION