from typing import Dict, Union
import numpy as np
from svm import SVM
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt


def vasualize_decision_region(svm: SVM, data: np.array, label: np.array) -> None:
    plot_decision_region(svm.model, data, label)
    plt.show()