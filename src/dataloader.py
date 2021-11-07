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
            Load image data

            Args:
                pth (str): the image file path

            Returns:
                np.array: the image data with shape (sample_num, 28, 28)
            """
            with open(pth, 'rb') as source:                                             # Open the dataset file
                magic, num, rows, cols = struct.unpack('>IIII', source.read(16))        # Unpack the file and get some dataset info
                imgs = np.fromfile(source, dtype=np.uint8).reshape(num, 28, 28)         # Using fromfile get the image data
            return imgs                                                                 # Return images with shape (sample_num, 28, 28)

        def load_label(pth: str) -> np.array:
            """
            Load label data

            Args:
                pth (str): the label file path

            Returns:
                np.array: the label data with shape (sample_num, )
            """
            with open(pth, 'rb') as source:                                             # Open file
                magic, num = struct.unpack('>II', source.read(8))                       # Unpack the file and get some dataset info
                labels = np.fromfile(source, dtype=np.uint8)                            # Using fromfile get the label data
            return labels                                                               # Return the labels with shape (sample_num, )
        
        self.train_img = load_img(self.train_pth + 'train-images-idx3-ubyte')           # Read train_set image data using function above
        self.train_label = load_label(self.train_pth + 'train-labels-idx1-ubyte')       # Read train_set label data using function above
        self.test_img = load_img(self.test_pth + 't10k-images-idx3-ubyte')              # Read test_set image data using function above
        self.test_label = load_label(self.test_pth + 't10k-labels-idx1-ubyte')          # Read test_set label data using function above
        
    def preprocess(self, inverse_color: bool = False, normalize: bool = True) -> None:
        """
        Image preprocessing including normalization and color_inverse

        Args:
            inverse_color (bool, optional): [Whether to use the color_inverse to all images]. Defaults to False.
            normalize (bool, optional): [Whether to use normalization to all images]. Defaults to True.
        """
        if inverse_color:
            self.train_img = 255. - self.train_img                                      # If using inverse_color function, transform the pixel x to 255 - x
            self.test_img = 255. - self.test_img                                        # Same to test_set images
        if normalize:
            self.train_img = self.train_img / 255.                                      # If using normalization function, transform the pixel x to x / 255, make every pixel data in [0, 1]
            self.test_img = self.test_img / 255.                                        # Same to test_set images
        
        self.train_img = self.train_img.reshape(-1, 28*28)                              # Transform the shape (sample_num, 28, 28) to (sample, 28*28)
        self.test_img = self.test_img.reshape(-1, 28*28)                                # Same to test_set images