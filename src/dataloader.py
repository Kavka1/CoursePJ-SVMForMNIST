from os import confstr, path
import struct
from typing import List, Dict, Tuple, Union
import numpy as np
from utils import load_config


class DataLoader(object):
    def __init__(self, config: Dict) -> None:
        super().__init__()

        self.dataset_pth = config['dataset_pth']
        self.train_pth = self.dataset_pth + "train/"
        self.test_pth = self.dataset_pth + "test/"

        self.train_img = None
        self.train_label = None
        self.test_img = None
        self.test_label = None

        self.load_data()

    def load_data(self) -> None:
        def load_img(pth: str) -> np.array:
            with open(pth, 'rb') as source:
                magic, num, rows, cols = struct.unpack('>IIII', source.read(16))
                imgs = np.fromfile(source, dtype=np.uint8).reshape(num, 28, 28)
            return imgs

        def load_label(pth: str) -> np.array:
            with open(pth, 'rb') as source:
                magic, num = struct.unpack('>II', source.read(8))
                labels = np.fromfile(source, dtype=np.uint8)
            return labels
        
        self.train_img = load_img(self.train_pth + 'train-images-idx3-ubyte')
        self.train_label = load_label(self.train_pth + 'train-labels-idx1-ubyte')
        self.test_img = load_img(self.test_pth + 't10k-images-idx3-ubyte')
        self.test_label = load_label(self.test_pth + 't10k-labels-idx1-ubyte')

        print("Load dataset over!")
        
    def preprocess(self):
        raise NotImplementedError("Wait for it")


if __name__ == "__main__":
    config = load_config()
    loader = DataLoader(config)