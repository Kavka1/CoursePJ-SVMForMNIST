from typing import List, Union, Dict
import json
import numpy as np
import struct
import os
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt


CONFIG_PATH = "/Users/xukang/Project/Repo/svm-minist/src/config.json"


def load_config(path: str = CONFIG_PATH):
    with open(path, mode='r') as source:
        config = json.load(source)
    return config


def preprocess_imgs(imgs: np.array) -> np.array:
    return imgs.astype(np.float64) / 255


def visualization(imgs, true_labels = None, model_path = "/Users/xukang/Project/Repo/svm-minist/model/svm.model"):
    clf = joblib.load(model_path)
    pred = clf.predict(imgs)

    for i, img in enumerate(imgs[:8]):
        plt.figure()
        plt.title(f"label: {true_labels[i]} pred: {pred_labels[i]}")
        plt.imshow(imgs[0].reshape((28, 28)))
        plt.show()


if __name__ == "__main__":
    train_set, train_label, test_set, test_label = load_dataset()
    train_set = preprocess_imgs(train_set)
    test_set = preprocess_imgs(test_set)

    
