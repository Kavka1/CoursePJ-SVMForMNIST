from typing import List, Dict, Union
from copy import copy
from dataloader import DataLoader
from utils import load_config
from svm import SVM
from multiprocessing import Process


def train(config: Dict):
    """
    Train function which open one experiment, witch will be called by Process to implement the parallel experiments

    Args:
        config (Dict): [the congit dict]
    """
    svm = SVM(config)                                               # Instantiate the SVM wrap 
    loader = DataLoader(config)                                     # Instantiate the data loader

    loader.preprocess(
        inverse_color=config['using_inverse_color'], 
        normalize=config['using_normalize'] 
    )                                                               # Preprocess the train_set and test_set images

    svm.train(
        data= loader.train_img[:config['train_sample_num']], 
        label= loader.train_label[:config['train_sample_num']]
    )                                                               # Train the model

    svm.evaluate_and_save(
        data= loader.test_img[:5000],
        label= loader.test_label[:5000]
    )                                                               # Evaluate the model, save model and report


def run() -> None:
    """
    Main function to run the parallel experiments based on the congfig.json setting
    """
    origin_config = load_config()                                   # Read the config.json file
    
    sample_num = origin_config['train_sample_num']                  # Get all sample_num parameters
    kernel = origin_config['kernel']                                # Get all kernel parameters
    inverse_color = origin_config['using_inverse_color']            # Get all inverse_color parameters 
    normalize = origin_config['using_normalize']                    # Get all normalization parameters

    for sample_n in sample_num:                                     # Do the grid search to get each combination of the parameters above
        for k in kernel:
            for inverse in inverse_color:
                for normal in normalize:
                    exp_name = f"sample-{sample_n}_kernel-{k}_inverse-{inverse}_normal-{normal}"        # Using the parameter combination as the experiment name and direction
                    config = copy(origin_config)                                                        # Copy the origin config dict
                    config.update({
                        'train_sample_num': sample_n,
                        'kernel': k,
                        'using_inverse_color': inverse,
                        'using_normalize': normal,
                        'save_path': origin_config['save_path'] + f'{exp_name}/'
                    })                                                                                  # Update the each parameter of the current experiment
                    p = Process(target= train, args=(config, ))                                         # Using the Process() to wrap the experiment
                    p.start()                                                                           # Start the experiment


if __name__ == "__main__":
    run()                                                           # Main function