from typing import List, Union, Dict
import json
import numpy as np
import struct
import os
import matplotlib.pyplot as plt


def load_config():
    config_path = os.path.abspath('.') + '/config.json'
    with open(config_path, mode='r') as source:
        config = json.load(source)
    return config


def check_path(path: str) -> None:
    if not os.path.exists(path):
        os.mkdir(path)


def visualization(imgs, true_labels = None, model_path = "/Users/xukang/Project/Repo/svm-minist/model/svm.model"):
    clf = joblib.load(model_path)
    pred = clf.predict(imgs)

    for i, img in enumerate(imgs[:8]):
        plt.figure()
        plt.title(f"label: {true_labels[i]} pred: {pred_labels[i]}")
        plt.imshow(imgs[0].reshape((28, 28)))
        plt.show()
