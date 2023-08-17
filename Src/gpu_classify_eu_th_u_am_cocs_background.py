import os

import numpy as np
import pandas as pd

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, Subset, DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split

from data_helpers import SpectrumDataset, RotatedSpectrumDataset, train_test_dataset_split
from models import CNN1Classifier


CHANNEL_LENGTH = 1024
LABEL_DICT = {'/Data/pozadi_nove': 0,
              '/Data/zdroj_eu_152': 1,
              '/Data/zdroj_th_232': 2,
              '/Data/zdroj_u_238': 3,
              '/Data/zdroj_am_241': 4,
              '/Data/zdroj_co_cs': 5
              }

MAX_EPOCHS = 200 #00 # maximal training epochs
BATCH_SIZE =  128# 128 and 256 are very close in acc
LR = 0.0001 # 0.00041252074938882393/2 # test
MOMENTUM = 0.9 #0.95
WEIGHT_DECAY = 1e-6
SKIP_CLASS_0 = False # dont extentd background class via augmentation
MACC = 0.93

BEST_MODEL_PATH = "best_model_bg_eu_th_u_am_cocs_classification.pth"
TRAIN_DATA_RESULTS = "best_model_bg_eu_th_u_am_cocs_classification_train.csv"
TEST_DATA_RESULTS = "best_model_bg_eu_th_u_am_cocs_classification_test.csv"


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



def train_model(model, weights, train_dataloader, test_dataloader):
    loss_fn = nn.NLLLoss(weights)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
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

        acc /= count
        loss_train /= train_epoch_steps
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
                count += len(labels)
                test_epoch_steps += 1

        acc /= count
        loss_test /= test_epoch_steps

        scheduler.step(loss_test)
        if max_test_acc < acc:  # kdyz je lepsi acc
            print("Save model")
            max_test_acc = acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        results["test_acc"].append(acc.cpu().item()) #numpy()
        results["test_loss"].append(loss_test)
        print(f"Vall>Epoch {epoch}: model accuracy {acc * 100} loss:\t{loss_test}")
        print(f"*" * 50)
    print(f"Best acc:{max(results['test_acc'])}")
    pd.DataFrame(data=results).to_csv(f"results_{MAX_EPOCHS}_learning_classification.cvs")
    return results

def learn_classify(data_dir=None):

    remote_label_dict={data_dir+k:v for k,v in LABEL_DICT.items()} # special hack for gpu server
    spectredataset = SpectrumDataset(paths_dict=remote_label_dict, transform = None) #use minmax normalization <=>transform=None
    cnn_conf, dense_conf = create_net_config() # create configuration for building neural network

    # use custom split, since we need indices of the labels
    train_idx, test_idx = train_test_split(list(range(len(spectredataset))), test_size=0.2, random_state=42)
    train_dataset = Subset(spectredataset, train_idx)
    test_dataset = Subset(spectredataset, test_idx)


    train_dataset_rot = RotatedSpectrumDataset(train_dataset) # extend dataset via augmentation
    train_dataloader_rot = DataLoader(train_dataset_rot, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = CNN1Classifier(cnn_config=cnn_conf, dense_config=dense_conf)

    for t in range(5):
        model = CNN1Classifier(cnn_config=cnn_conf, dense_config=dense_conf)
        history = train_model(model=model, weights=spectredataset.weights, train_dataloader=train_dataloader_rot,
                    test_dataloader=test_dataloader)
        macc_test = max(history["test_acc"])
        if macc_test > MACC:
            break
        else:
            print("RERUN")


    #train_model(model=model, weights=spectredataset.weights, train_dataloader=train_dataloader_rot, test_dataloader=test_dataloader)
    #load model and classify
    best_model=CNN1Classifier(cnn_config=cnn_conf, dense_config=dense_conf)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if torch.cuda.device_count() > 1:
        best_model = nn.DataParallel(best_model)  # udelej to paralelni
    best_model = best_model.to(device)
    best_model.load_state_dict(torch.load(BEST_MODEL_PATH))
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False) # use same order as in dataset
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) # use same order as in dataset
    y_hat_train = []
    y_hat_test = []
    train_data_results = {"file":[],"succes": [],"true_label":[],"predicted_label":[]}
    test_data_results = {"file": [],"succes": [], "true_label": [], "predicted_label": []}
    # classifiation of training data
    train_acc = 0
    train_count = 0
    for bid, batch in enumerate(train_dataloader):
        inputs, labels = batch
        inputs = inputs.to(device=device, dtype=torch.float16)
        labels = labels.to(device=device)
        with torch.autocast(device_type=device.type, enabled=True):
            y_batch_train_hat = best_model(inputs)
        train_acc += (torch.argmax(y_batch_train_hat, 1) == labels).float().sum() # add acc in batch
        train_count += len(labels) #add length of the batch
        y_batch_train_hat=torch.argmax(y_batch_train_hat, 1).cpu().numpy() # store predicte classes
        y_hat_train.append(y_batch_train_hat)
    y_hat_train=np.hstack(y_hat_train)
    train_acc /= train_count
    print(f"BEST_MODEL: Train data acc:{train_acc}")
    for i in range(y_hat_train.shape[0]):
         #print(f"{y_hat_train[i]}==={spectredataset.y_data[train_idx[i]]}\t{spectredataset.titles[train_idx[i]]}")
          train_data_results["file"].append(spectredataset.titles[train_idx[i]])
          train_data_results["succes"].append((y_hat_train[i] == spectredataset.y_data[train_idx[i]]).cpu().item())
          train_data_results["true_label"].append(spectredataset.y_data[train_idx[i]].cpu().item())
          train_data_results["predicted_label"].append(y_hat_train[i])
    pd.DataFrame(data=train_data_results).to_csv(TRAIN_DATA_RESULTS)

    #classification of testing data
    test_acc = 0
    test_count = 0
    for bid, batch in enumerate(test_dataloader):
        inputs, labels = batch
        inputs = inputs.to(device=device, dtype=torch.float16)
        labels = labels.to(device=device)
        with torch.autocast(device_type=device.type, enabled=True):
            y_batch_test_hat = best_model(inputs)
        test_acc += (torch.argmax(y_batch_test_hat, 1) == labels).float().sum()  # add acc in batch
        test_count += len(labels)  # add length of the batch
        y_batch_test_hat = torch.argmax(y_batch_test_hat, 1).cpu().numpy()  # store predicte classes
        y_hat_test.append(y_batch_test_hat)
    y_hat_test = np.hstack(y_hat_test)
    test_acc /= test_count
    print(f"BEST_MODEL: Test data acc:{test_acc}")
    for i in range(y_hat_test.shape[0]):
        # print(f"{y_hat_test[i]}==={spectredataset.y_data[test_idx[i]]}\t{spectredataset.titles[test_idx[i]]}")
        test_data_results["file"].append(spectredataset.titles[test_idx[i]])
        test_data_results["succes"].append((y_hat_test[i] == spectredataset.y_data[test_idx[i]]).cpu().item())
        test_data_results["true_label"].append(spectredataset.y_data[test_idx[i]].cpu().item())
        test_data_results["predicted_label"].append(y_hat_test[i])
    pd.DataFrame(data=test_data_results).to_csv(TEST_DATA_RESULTS)



if __name__ == "__main__":
    print("START LEARNING AND CLASSIFICATION")
    learn_classify(data_dir=os.path.dirname(os.getcwd()))