import os
import json
from functools import partial

import numpy as np
import pandas as pd

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, Subset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import  compute_class_weight


from data_helpers import SpectrumDataset,  RotatedSpectrumDataset
from models import CNN1Classifier


CHANNEL_LENGTH = 1024

# dictionary with paths
LABEL_DICT = {'/Data/zdroj_co': 0,
              '/Data/zdroj_cs': 1,
              '/Data/zdroj_eu_152': 2,
              '/Data/zdroj_th_232': 3,
              '/Data/zdroj_u_238': 4,
              '/Data/zdroj_am_241': 5,
              '/Data/zdroj_co_cs': 6
              }
# maximal training epochs
MAX_EPOCHS = 200
BATCH_SIZE =  256 # 128 nejlepsi yatim
P_ROTATION = 0.0
LR =  0.00041252074938882393/2 # test
MOMENTUM = 0.9 #0.95
WEIGHT_DECAY = 1e-6

# constant for cross validation

SPLITS = 5


def kfold_validation(data_dir=None):

    remote_label_dict={data_dir+k:v for k,v in LABEL_DICT.items()} # special hack for gpu server
    spectredataset = SpectrumDataset(paths_dict=remote_label_dict, transform = None) #use minmax normalization <=>transform=None

    skf = StratifiedKFold(n_splits=SPLITS)

    for fold_index, (train_index, test_index) in enumerate(skf.split(spectredataset.X_data,spectredataset.y_data)):
        print(f'FOLD: {fold_index}')
        fold_class_weight = compute_class_weight(class_weight="balanced", y=spectredataset.y_data[train_index].numpy(), classes=[i for i in LABEL_DICT.values()])
        print(f"CLASS WIGHTS: {np.array2string(fold_class_weight, max_line_width=100)}")
        train_dataset, test_dataset = Subset(spectredataset, train_index), Subset(spectredataset, test_index)
        train_dataset = RotatedSpectrumDataset(train_dataset) # extaned training data via augmentation
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


    print("DONE")




if __name__=="__main__":
    print("START KFOLD VALIDATION")
    print(f"USE:{SPLITS} FOLDS")
    kfold_validation(data_dir=os.path.dirname(os.getcwd()))