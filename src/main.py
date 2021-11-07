from typing import List, Dict, Union
from copy import copy
from dataloader import DataLoader
from utils import load_config
from svm import SVM
from multiprocessing import Process


def train(config: Dict):
    svm = SVM(config)
    loader = DataLoader(config)

    loader.preprocess(
        inverse_color=config['using_inverse_color'], 
        normalize=config['using_normalize']
    )

    svm.train(
        data= loader.train_img[:config['train_sample_num']], 
        label= loader.train_label[:config['train_sample_num']]
    )
    svm.evaluate_and_save(
        data= loader.test_img[:5000],
        label= loader.test_label[:5000]
    )


def run() -> None:
    origin_config = load_config()
    
    sample_num = origin_config['train_sample_num']
    kernel = origin_config['kernel']
    inverse_color = origin_config['using_inverse_color']
    normalize = origin_config['using_normalize']

    for sample_n in sample_num:
        for k in kernel:
            for inverse in inverse_color:
                for normal in normalize:
                    exp_name = f"sample-{sample_n}_kernel-{k}_inverse-{inverse}_normal-{normal}"
                    config = copy(origin_config)
                    config.update({
                        'train_sample_num': sample_n,
                        'kernel': k,
                        'using_inverse_color': inverse,
                        'using_normalize': normal,
                        'save_path': origin_config['save_path'] + f'{exp_name}/'
                    })
                    p = Process(target= train, args=(config, ))
                    p.start()


if __name__ == "__main__":
    run()