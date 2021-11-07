from typing import List, Dict, Union
from sklearn import svm
import joblib
import numpy as np
import os, sys, time
from utils import load_dataset, preprocess_imgs, visualization
import matplotlib.pyplot as plt


def create_svm(decision = 'ovr'):
    clf = svm.SVC(C=1.0, kernel='rbf', decision_function_shape=decision)
    return clf


def train(data, label, path = "/Users/xukang/Project/Repo/svm-minist/model/svm.model"):
    clf = create_svm()
    st = time.time()
    rf = clf.fit(data, label)
    joblib.dump(rf, path)
    et = time.time()
    print(f"Traning time : {et - st}")


def evaluate(data, label, model_path = "/Users/xukang/Project/Repo/svm-minist/model/svm.model"):
    clf = joblib.load(model_path)
    pred = clf.predict(data)
    score = clf.score(data, label)
    print(f"Precise: {score}")
    return pred


if __name__ == "__main__":
    train_set, train_label, test_set, test_label = load_dataset()
    train_set = preprocess_imgs(train_set)
    test_set = preprocess_imgs(test_set)
    
    pred_label = evaluate(test_set, test_label)

    visualization(test_set, test_label, pred_label)