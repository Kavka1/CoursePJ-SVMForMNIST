from typing import List, Dict, Tuple, Union, Callable
import numpy as np
import time
import pickle
import json
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from kernel import Histogram_Intersection_Kernel
from utils import check_path


class SVM(object):
    """
    Wrap of sklearn.svm.SVC, including some base operation like traning, evaluation, save_model. etc
    """
    def __init__(self, config: Dict) -> None:
        """
        Initialize

        Args:
            config (Dict): [the config dict involving some setting]
        """
        super().__init__()                                                          # Super class initialize

        self.save_path = config['save_path']                                        # Read the save path setting
        check_path(self.save_path)                                                  # Check whether this path exists, if not then create it 

        self.kernel = self._judge_kernel(config['kernel'])                          # Using the _judge_kernel() to read the kernel setting
        self.model = SVC(kernel=self.kernel, gamma='auto')                          # Instantiate the SVC model

    def _judge_kernel(self, kernel_name: str) -> Union[Callable, str]:
        """
        Judge which type of kernel will be used

        Args:
            kernel_name (str): [the kernel string in confit dict]

        Raises:
            ValueError: [if the kernel name not reasonable, then the error raises]

        Returns:
            Union[Callable, str]: [the kernel function (user defined) or the kernel name string (already exists)]
        """
        if kernel_name == "Histogram":                                              # If use the histogram kernel, then use the function defined in kernel.py
            kernel = Histogram_Intersection_Kernel
        elif kernel_name in ['rbf', 'poly', 'linear', 'sigmoid']:                   # If use the four built-in kernel, then just return the kernel string
            kernel = kernel_name
        else:                                                                       # If the kernel name not reasonable, then raise error
            raise ValueError(f"No such kind kernel: {config['kernel']}")            
        return kernel                                                               # Return the kernel string or Callable function

    def train(self, data: np.array, label: np.array) -> None:
        """
        Train the SVC model using the train_set

        Args:
            data (np.array): [train_set images data with shape (sample_num, 28*28)]
            label (np.array): [train_set labels data with shape (sample_num, )]
        """
        print("-------- Start training --------")                                   # Print log
        start = time.time()                                                         # Count the start time
        self.model.fit(data, label)                                                 # Fit the train data
        print("--------- Training over --------")                                   # Print log
        print(f"Training takes time {time.time() - start}")                         # Print log about train time

    def evaluate(self, data: np.array, label: np.array) -> None:
        """
        Evaluate the model

        Args:
            data (np.array): [image data with shape (sample_num, 28*28)]
            label (np.array): [label data with shape (sample_num, )]
        """
        pred = self.model.predict(data)                                             # Make the prediction using the trained model
        print(f"Classification Report: \n{classification_report(label, pred)}")     # Print logger with the output report

    def evaluate_and_save(self, data: np.array, label: np.array) -> None:
        """
        Evaluate the model and save the report with model

        Args:
            data (np.array): [image data with shape (sample_num, 28*28)]
            label (np.array): [label data with shape (sample_num, )]
        """
        pred = self.model.predict(data)                                             # Make the prediction using the trained model
        report = classification_report(label, pred, output_dict=True)               # Get the prediction report
        
        self.save_model()                                                           # Save model file
        self.save_report(report)                                                    # Save report file

    def save_report(self, report: Dict) -> None:
        """
        Save the prediction report which has some data analyze

        Args:
            report (Dict): [the report dictionary]
        """
        with open(self.save_path + 'result.json', 'w') as f:                            # Open the file 
            json.dump(report, f)                                                        # Using json.dump() to transform the dict to json type
        print(f"----- Save result report to {self.save_path + 'result.json'} -----")    # Print log

    def save_model(self) -> None:
        """
        Save model as pickle file
        """
        model_path = self.save_path + f'model.pickle'                               # Get the exact save path 

        with open(model_path, 'wb') as f:                                           # Open the file with binary write mode
            pickle.dump(self.model, f)                                              # Save the model file
        print(f"------ Model saved to {model_path} -------")                        # Print log

    def load_model(self) -> None:
        """
        Load model from pickle file
        """
        model_path = self.save_path + f'model.pickle'                               # Get the exact file path
        
        with open(model_path, 'rb') as f:                                           # Open the file with binary read mode
            self.model = pickle.load(f)                                             # Load the model
        print(f"------ Loaded model from {model_path} -------")                     # Print log