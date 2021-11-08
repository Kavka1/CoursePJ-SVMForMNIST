from json import load
from typing import Dict, Union
import numpy as np
import pickle
from scipy.sparse import data
from sklearn.svm import SVC
from dataloader import DataLoader
from svm import SVM
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from utils import load_config


def show_test_images(images: np.array, labels: np.array, pred_labels: np.array, num_label: int = 10, num_img: int = 3) -> None:
    """
    Show the images from test_set with their true labels and pred labels

    Args:
        images (np.array): [image data with shape (num_samples, 28*28)]
        labels (np.array): [true label data with shape (num_samples, )]
        pred_labels (np.array): [pred label data with shape (num_samples, )]
        num_label (int, optional): [the labels num need to show]. Defaults to 10.
        num_img (int, optional): [num of image for each label]. Defaults to 3.
    """
    idxs = [np.random.choice(np.where(labels == label)[0], size = num_img) for label in range(np.max(labels) + 1)]      # Choose three images for each label, and get the index
    imgs = [images[idx].reshape(3, 28, 28) for idx in idxs]                                                             # Get the images based on the index
    preds = [pred_labels[idx] for idx in idxs]                                                                          # Get the pred labels based on the index
    fig = plt.figure(figsize=(10,15))                                                                                   # Create a plot with 10*15 images
    plt.ion()                                                                                                           # Open the interactive mode
    for label in range(num_label):                                                                  # For each label
        for idx in range(num_img):                                                                  # For each image in the label
            ax = fig.add_subplot(num_label , num_img, label * 3 + idx + 1)                          # Add the image to the cressponding subplot
            ax.imshow(imgs[label][idx], cmap = 'gray')                                              # Show image in gray mode
            ax.set_title(f'label:{label},pred:{preds[label][idx]}')                                 # Set title
            ax.set_xticks([])                                                                       # Remove the x axis                                        
            ax.set_yticks([])                                                                       # Remove the y axis

    plt.tight_layout()                                                                              # Show images more tight
    plt.ioff()                                                                                      # Close the interactiva mode
    plt.show()                                                                                      # Show images


def show_report(labels: np.array, pred: np.array) -> None:
    """
    Show the train model performance report

    Args:
        labels (np.array): [True labels of the images]
        pred (np.array): [Predication of the images]
    """
    print(f"Performance report: \n {classification_report(labels, pred)}")                          # Print log and report



if __name__ == "__main__":
    model = SVC(kernel='linear')
    data_loader = DataLoader(load_config())
    data_loader.preprocess()
    model.fit(data_loader.train_img[:10000], data_loader.train_label[:10000])

    test_sample = data_loader.test_img[:1000]
    test_label = data_loader.test_label[:1000]
    pred = model.predict(test_sample)

    show_test_images(test_sample, test_label, pred)
    #show_report(test_label, pred)