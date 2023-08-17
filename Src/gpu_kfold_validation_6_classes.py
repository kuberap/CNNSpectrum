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
from sklearn.metrics import classification_report,confusion_matrix


from data_helpers import SpectrumDataset,  RotatedSpectrumDataset
from models import CNN1Classifier
import classification_reporter as reporter

CHANNEL_LENGTH = 1024

# dictionary with paths
LABEL_DICT = {'/Data/zdroj_co': 0,
              '/Data/zdroj_cs': 1,
              '/Data/zdroj_eu_152': 2,
              '/Data/zdroj_th_232': 3,
              '/Data/zdroj_u_238': 4,
             # '/Data/zdroj_am_241': 5,
              '/Data/zdroj_co_cs': 5
              }

MAX_EPOCHS = 200 # maximal training epochs
BATCH_SIZE =  128 # 128 and 256 are very close in acc
LR = 0.0001 # 0.00041252074938882393/2 # test
MOMENTUM = 0.9 #0.95
WEIGHT_DECAY = 1e-6

# constant for cross validation
SPLITS = 5

# report parameters
TARGET_NAMES = ['CO', 'CS', 'EU', 'TH', 'U', 'COCS']  # class names for report
NUM_CLASSES = len(TARGET_NAMES)
TASK_NAME = 'SOURCE_DIFFERENTIATE_CO_CS_EU_TH_U_COCS'
TASK_DESCRIPTION = 'SOURCES CO_CS_EU_TH_U_COCS LABELS 0,1,2,3,4,5,.'
REPORT_OUTPUT_PATH = f'../Results/report-{TASK_NAME}.txt'  # zde je ulozena statistika ulohy
RESULTS_OUTPUT_PATH = f'../Results/result-'  # zde jsou vysledky jednotlivych volani



def compute_flatten_size(cnn_config, input_length=1024, verbose=False):
    if verbose:
        print(f"input length:{input_length}")
    out_channels=None
    for i,conf in enumerate(cnn_config):
       in_channels=conf["in_channels"]
       out_channels=conf["out_channels"]
       kernel_size=conf["kernel_size"]
       pool_size=conf["pool_size"]
       input_length=input_length-(kernel_size//2)*2
       if verbose:
            print(f"cnn1 - block:{i}: size:{input_length} x units:{out_channels}")
       input_length = input_length - (kernel_size // 2)*2
       if verbose:
            print(f"cnn2- block:{i}: size:{input_length} x units:{out_channels}")
       input_length//=pool_size
       if verbose:
            print(f"final- block:{i}: size:{input_length} x units:{out_channels}")
    if verbose:
        print(f"Flatten units:{input_length*out_channels}")
    return input_length*out_channels



def create_net_config():
    # {'cnn1-out': 16, 'cnn1-kernel': 5, 'cnn1-dropout': 0.001, 'cnn2-out': 128, 'cnn2-kernel': 3,
    #          'cnn2-dropout': 0.01, 'dense1_out': 1024, 'dense1-dropout': 0.001, 'batch_size': 32,
    #          'lr': 0.00041252074938882393, 'momentum': 0.95, 'p-rotation': 0.3, 'val_split': 0.1, 'max_epochs': 40}

    cnn_conf = [
        {"in_channels": 1,
         "out_channels": 32,
         "kernel_size": 3,
         "pool_size": 2,  # division by 2 is universal
         "dropout":  0.001
         },
        {"in_channels": 32,  # must be the same as the output from the layer one
         "out_channels": 128,
         "kernel_size": 3,
         "pool_size": 2,  # division by 2 is universal
         "dropout": 0.001
         }
    ]
    flatten_size = compute_flatten_size(cnn_conf)
    dense_conf = [
        {"in": flatten_size,
         "out": 256,
         "activation": True,
         "dropout": 0.05,
         },

        {
            "in": 256,# must be the same as the output from the layer one
            "out": len(LABEL_DICT),
            "activation": False,
            "dropout": None
         }
    ]

    return cnn_conf, dense_conf


def train_model(model, weights, train_dataloader, test_dataloader, fold):
    loss_fn = nn.NLLLoss(weights)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    y_hat_train_best_classes = None
    y_hat_test_best_classes = None
    # storing actual order of labels due to shuffling in each epoch, we need to store real labels
    y_train_best_classes = None
    y_test_best_classes = None

    print(f"USE CUDA: {use_cuda} ")
    if use_cuda:
        print(f"CUDA devices: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)  # udelej to paralelni
            print(f"Model set to data parallel model.")
        model = model.to(device)  # hod to na GPU
        loss_fn = loss_fn.to(device)  # hodn na GPU i kriterialni funkci

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)  # lr=0.0001
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)
    print("Start - learning")
    results = {
        "train_loss": [], "test_loss": [],
        "train_acc": [], "test_acc": []}
    max_test_acc = 0.0
    for epoch in range(MAX_EPOCHS):
        loss_train = 0.0
        acc = 0
        count = 0
        train_epoch_steps = 0
        y_hat_train_classes = []
        y_hat_test_classes = []
        y_train_classes = []
        y_test_classes = []

        for bid, batch in enumerate(train_dataloader):
            inputs, labels = batch
            inputs = inputs.to(device=device, dtype=torch.float16)
            labels = labels.to(device=device)
            with torch.autocast(device_type=device.type, enabled=True):
                y_pred = model(inputs)
                loss = loss_fn(y_pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            item = loss.item()
            loss_train += item
            # print(f"\tEPOCH:{epoch}:{bid}--->{item}::::{loss_train}")
            acc += (torch.argmax(y_pred, 1) == labels).float().sum()
            count += len(labels)
            train_epoch_steps += 1
            y_hat_train_classes.append(torch.argmax(y_pred, 1).cpu().numpy())
            y_train_classes.append(labels.cpu().numpy())
        acc /= count
        loss_train /= train_epoch_steps
        y_hat_train_classes = np.hstack(y_hat_train_classes)
        y_train_classes = np.hstack(y_train_classes)
        results["train_acc"].append(acc.cpu().item())
        results["train_loss"].append(loss_train)
        print(f"Train>Epoch {epoch}/{MAX_EPOCHS}: model accuracy {acc * 100} loss:\t{loss_train}")

        acc = 0
        count = 0
        loss_test = 0.0
        test_epoch_steps = 0
        for inputs, labels in test_dataloader:
            with torch.no_grad():
                inputs = inputs.to(device=device, dtype=torch.float16)
                labels = labels.to(device=device)
                with torch.autocast(device_type=device.type, enabled=True):
                    y_pred = model(inputs)
                    loss = loss_fn(y_pred, labels)
                item = loss.item()
                loss_test += item
                acc += (torch.argmax(y_pred, 1) == labels).float().sum()
                y_hat_test_classes.append(torch.argmax(y_pred, 1).cpu().numpy())
                y_test_classes.append(labels.cpu().numpy()) # numpy()
                count += len(labels)
                test_epoch_steps += 1

        acc /= count
        loss_test /= test_epoch_steps
        y_hat_test_classes = np.hstack(y_hat_test_classes)
        y_test_classes = np.hstack(y_test_classes)
        scheduler.step(loss_test)
        if max_test_acc < acc:  # kdyz je lepsi acc
            y_hat_train_best_classes = y_hat_train_classes
            y_hat_test_best_classes = y_hat_test_classes

            y_train_best_classes = y_train_classes
            y_test_best_classes = y_test_classes
            print("Save model")
            max_test_acc = acc
            torch.save(model.state_dict(), f"best_model_.pth")
        results["test_acc"].append(acc.cpu().item()) #numpy()
        results["test_loss"].append(loss_test)
        print(f"Vall>Epoch {epoch}: model accuracy {acc * 100} loss:\t{loss_test}")
        print(f"*" * 50)
    print(f"Best acc:{max(results['test_acc'])}")
    pd.DataFrame(data=results).to_csv(f"results_{MAX_EPOCHS}_fold_{fold}.cvs")
    return y_hat_train_best_classes, y_hat_test_best_classes,y_train_best_classes,y_test_best_classes, results


def kfold_validation(data_dir=None):
    # initialization of temporary array
    c_matrices_train = []
    c_matrices_test = []
    reports_train = []
    reports_test = []

    remote_label_dict={data_dir+k:v for k,v in LABEL_DICT.items()} # special hack for gpu server
    spectredataset = SpectrumDataset(paths_dict=remote_label_dict, transform = None) #use minmax normalization <=>transform=None
    cnn_conf, dense_conf = create_net_config() # create configuration for building neural network
    skf = StratifiedKFold(n_splits=SPLITS)

    # ------------ priprava reportu dat------------------
    report_data = {'TASK': TASK_NAME, 'DESCRIPTION': TASK_DESCRIPTION}
    report_data['LABEL_DATA_PATH'] = LABEL_DICT  # zapis i cesty a i pouzite labely
    labels, label_count = np.unique(spectredataset.y_data.numpy(), return_counts=True)
    report_data['DATA_INFO'] = (TARGET_NAMES, labels, label_count)  # labely, pocty, jmena
    report_data['LOAD_TOTAL_LABELS'] = len(spectredataset)  # celkovy pocet prikladu
    reporter.report_experiment(state='init', data_dict=report_data, path=REPORT_OUTPUT_PATH)

    for fold_index, (train_index, test_index) in enumerate(skf.split(spectredataset.X_data,spectredataset.y_data)):
        print(f'FOLD: {fold_index}')
        fold_class_weight = compute_class_weight(class_weight="balanced", y=spectredataset.y_data[train_index].numpy(), classes=labels) #classes=[i for i in LABEL_DICT.values()]#
        print(f"CLASS WEIGHTS: {np.array2string(fold_class_weight, max_line_width=100)}")
        train_dataset, test_dataset = Subset(spectredataset, train_index), Subset(spectredataset, test_index)
        train_dataset = RotatedSpectrumDataset(train_dataset) # extaned training data via augmentation
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

        model = CNN1Classifier(cnn_config=cnn_conf, dense_config=dense_conf)
        y_hat_train, y_hat_test,y_train, y_test, history = train_model(model=model,weights=torch.from_numpy(fold_class_weight).float(), train_dataloader=train_dataloader, test_dataloader=test_dataloader, fold=fold_index)

        print(classification_report(y_test, y_hat_test, target_names=TARGET_NAMES))
        print(confusion_matrix(y_test, y_hat_test))


        # --------------- produkce vysledku pro report --------------------
        report_data['FOLD'] = fold_index  # zapis si index
        report_data['HISTORY'] = history  # historie uceni
        report_data['TRAIN_LABELS'] = len(train_dataset)
        report_data['TEST_LABELS'] = len(test_dataset)
        labels_train, labels_train_count = np.unique(train_dataset.y_data.numpy(), return_counts=True)
        report_data['DATA_INFO_TRAIN'] = (TARGET_NAMES,  labels_train, labels_train_count)
        labels_test, labels_test_count = np.unique(train_dataset.y_data.numpy(), return_counts=True)
        report_data['DATA_INFO_TEST'] = (TARGET_NAMES, labels_test, labels_test_count)
        report_data['REPORT_TRAIN'] = classification_report(y_train, y_hat_train, target_names=TARGET_NAMES)
        report_data['REPORT_TEST'] = classification_report(y_test, y_hat_test, target_names=TARGET_NAMES)
        report_data['C_MATRIX_TEST'] = confusion_matrix(y_test, y_hat_test)
        report_data['C_MATRIX_TRAIN'] = confusion_matrix(y_train, y_hat_train)
        reporter.report_experiment(state='fold', data_dict=report_data, path=REPORT_OUTPUT_PATH)
        # --------------------------------------------------------------------------------

        # ----------------- logovani pro vypocet prumeru -------------------
        c_matrices_test.append(confusion_matrix(y_test, y_hat_test))
        c_matrices_train.append(confusion_matrix(y_train, y_hat_train))
        # udelej si seznam slovniku s reporty, viz output_dict=True
        reports_test.append(classification_report(y_test, y_hat_test, target_names=TARGET_NAMES, output_dict=True))
        reports_train.append(classification_report(y_train, y_hat_train, target_names=TARGET_NAMES, output_dict=True))
        del model
        # -------------- konec foldu----------------------

    C_average_test = np.array(c_matrices_test)
    C_average_train = np.array(c_matrices_train)
    report_data['C_MATRIX_AVG_TEST'] = C_average_test.mean(axis=0)  # prumer vsech c matic
    report_data['C_MATRIX_AVG_TRAIN'] = C_average_train.mean(axis=0)
    metrics = ['precision', 'recall', 'f1-score', 'support']
    total_scores = {}
    acc = []
    for d in reports_test:
        for target in TARGET_NAMES:  # pres vsechny tridy
            for metric in metrics:
                key = f'{target}:{metric}'
                if not key in total_scores:
                    total_scores[key] = []
                total_scores[key].append(d[target][metric])
        acc.append(d['accuracy'])

    report_data['TOTAL_SCORES'] = total_scores, acc
    reporter.report_experiment(state='end', data_dict=report_data, path=REPORT_OUTPUT_PATH)
    # --------- Vypis informaci na obrayovku
    for m in metrics:
        print(f'\t{m:10}(mean; std)', end="\t")
    for target in TARGET_NAMES:
        print(f'\n{target}', end="\t")
        for metric in metrics:
            key = f'{target}:{metric}'
            pole = np.array(total_scores[key])
            mean = pole.mean()
            std = pole.std()
            print(f'{mean:0.6f};{std:0.6f}', end="\t\t\t")
    acc = np.array(acc)
    print(f'\n\nACCURACY (mean;std): {acc.mean():0.6f};{acc.std():0.6f}')
    print("DONE")




if __name__=="__main__":
    print("START KFOLD VALIDATION")
    print(f"USE:{SPLITS} FOLDS")
    kfold_validation(data_dir=os.path.dirname(os.getcwd()))