from typing import List, Dict, Tuple, Union, Callable
import struct
import numpy as np
from numpy.lib.function_base import select
from utils import load_config


class DataLoader(object):
    """
    The wrap of dataset including load data and preprocessing images 
    """
    def __init__(self, config: Dict) -> None:
        """ Initialize

        Args:
            config (Dict): the experiment setting dictionary from config.json
        """
        super().__init__()

        self.dataset_pth = config['dataset_pth']            # The dataset path
        self.train_pth = self.dataset_pth + "train/"        # Get the train dataset path
        self.test_pth = self.dataset_pth + "test/"          # Get the test dataset path

        self.train_img = None                               # train_set img
        self.train_label = None                             # train_set label
        self.test_img = None                                # test_set img
        self.test_label = None                              # test_set label

        self.__load_data()                                  # load data

    def __load_data(self) -> None:
        """ 
        Load the dataset including images and labels
        """
        def load_img(pth: str) -> np.array:
            """
            Load images

            Args:
                pth (str): the image file path

            Returns:
                np.array: the image data with shape (sample_num, 28, 28)
            """
            with open(pth, 'rb') as source:                                             # Open the dataset file
                magic, num, rows, cols = struct.unpack('>IIII', source.read(16))        # 
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
        
    def preprocess(self, inverse_color: bool = False, normalize: bool = True) -> None:
        if inverse_color:
            self.train_img = 255. - self.train_img
            self.test_img = 255. - self.test_img
        if normalize:
            self.train_img = self.train_img / 255.
            self.test_img = self.test_img / 255.
        
        self.train_img = self.train_img.reshape(-1, 28*28)
        self.test_img = self.test_img.reshape(-1, 28*28)


if __name__ == "__main__":
    config = load_config()
    loader = DataLoader(config)