from typing import List, Dict, Tuple, Union, Callable
import numpy as np
import time
import pickle
import json
import os
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from kernel import Histogram_Intersection_Kernel
from utils import load_config, check_path
from dataloader import DataLoader


class SVM(object):
    def __init__(self, config: Dict) -> None:
        super().__init__()

        self.save_path = config['save_path']
        check_path(self.save_path)

        self.kernel = self._judge_kernel(config['kernel'])
        self.model = SVC(kernel=self.kernel, gamma='auto')

    def _judge_kernel(self, kernel_name: str) -> Union[Callable, str]:
        if kernel_name == "Histogram":
            kernel = Histogram_Intersection_Kernel
        elif kernel_name in ['rbf', 'poly', 'linear', 'sigmoid']:
            kernel = kernel_name
        else:
            raise ValueError(f"No such kind kernel: {config['kernel']}")
        return kernel

    def train(self, data: np.array, label: np.array) -> None:
        print("-------- Start training --------")
        start = time.time()
        self.model.fit(data, label)
        print("--------- Training over --------")
        print(f"Training takes time {time.time() - start}")

    def evaluate(self, data: np.array, label: np.array) -> None:
        pred = self.model.predict(data)
        print(f"Classification Report: \n{classification_report(label, pred)}")

    def evaluate_and_save(self, data: np.array, label: np.array) -> None:
        pred = self.model.predict(data)
        report = classification_report(label, pred, output_dict=True)
        
        self.save_model()
        self.save_report(report)

    def save_report(self, report: Dict) -> None:
        with open(self.save_path + 'result.json', 'w') as f:
            json.dump(report, f)
        print(f"----- Save result report to {self.save_path + 'result.json'} -----")

    def save_model(self) -> None:
        model_path = self.save_path + f'model.pickle'

        with open(model_path, 'wb') as f: #以二进制写的方式打开文件
            pickle.dump(self.model, f) 
        print(f"------ Model saved to {model_path} -------") #打印日志

    def load_model(self) -> None:
        model_path = self.save_path + f'model.pickle'
        
        with open(model_path, 'rb') as f: #以二进制读的方式打开文件
            self.model = pickle.load(f)
        print(f"------ Loaded model from {model_path} -------")