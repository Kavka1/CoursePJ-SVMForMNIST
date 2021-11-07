from typing import List, Union, Dict
import json
import os


def load_config() -> Dict:
    """
    Load the experiment config from file config.json

    Returns:
        Dict: [The config Dict like {'kernel': 'rbf', 'sample_num': 5000, ...}]
    """
    config_path = os.path.abspath('.') + '/config.json'                 # Get the config.json file path
    with open(config_path, mode='r') as source:                         # Open the config.json file
        config = json.load(source)                                      # Load the json file and transform it to Dict type 
    return config                                                       # Return the config dict


def check_path(path: str) -> None:
    """
    Check whether the path exists, if not then create the direction

    Args:
        path (str): [the path which needs check]
    """
    if not os.path.exists(path):                                        # Check whether this path exist
        os.mkdir(path)                                                  # if don't exists, then create this direction

