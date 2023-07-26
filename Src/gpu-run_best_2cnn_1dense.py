import os
import json
from functools import partial
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

from  run_7_classes_config import *

from data_helpers import SpectrumDataset, train_test_dataset_split, roll_batched_channels
from models import CNN1Classifier
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
         "out_channels": 16,
         "kernel_size": 5,
         "pool_size": 2,  # division by 2 is universal
         "dropout":  0.001
         },
        {"in_channels": 16,  # must be the same as the output from the layer one
         "out_channels": 128,
         "kernel_size": 3,
         "pool_size": 2,  # division by 2 is universal
         "dropout": 0.01
         }
    ]
    flatten_size = compute_flatten_size(cnn_conf)
    dense_conf = [
        {"in": flatten_size,
         "out": 1024,
         "activation": True,
         "dropout": 0.001,
         },
        {
            "in": 1024,# must be the same as the output from the layer one
            "out": len(LABEL_DICT),
            "activation": False,
            "dropout": None
         }
    ]

    return cnn_conf, dense_conf




def train_method(data_dir=None):
    # Before normalization:  Data set mean: tensor([4.8696]) Data set std: tensor([3.1731])

    # special hack
    remote_label_dict={data_dir+k:v for k,v in LABEL_DICT.items()}

    spectredataset = SpectrumDataset(paths_dict=remote_label_dict, transform=transforms.Normalize((4.8696), (3.1731)))
    train_dataset, test_dataset = train_test_dataset_split(spectredataset)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    cnn_conf, dense_conf = create_net_config()
    model = CNN1Classifier(cnn_config=cnn_conf, dense_config=dense_conf)

    loss_fn = nn.NLLLoss(spectredataset.weights)
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


    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)  # lr=0.0001

    start_epoch = 0

    print("Start - learning")
    results = {
        "train_loss": [], "test_loss": [],
        "train_acc": [], "test_acc": []}
    max_test_acc = 0.0
    for epoch in range(0, MAX_EPOCHS):
        loss_train = 0.0
        acc = 0
        count = 0
        train_epoch_steps=0
        for bid, batch in enumerate(train_dataloader):
            inputs, labels = batch
            if torch.rand(1) < P_ROTATION:
                inputs = roll_batched_channels(inputs, left=-2, right=2)

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
            train_epoch_steps+=1
        acc /= count
        loss_train/=train_epoch_steps
        results["train_acc"].append(acc.cpu().numpy())
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
                test_epoch_steps+=1

        acc /= count
        loss_test/=test_epoch_steps
        if max_test_acc < acc:  # kdyz je lepsi acc
            print("Save model")
            max_test_acc = acc
            torch.save(model.state_dict(), f"best_model_.pth")
        results["test_acc"].append(acc.cpu().numpy())
        results["test_loss"].append(loss_test)
        print(f"Vall>Epoch {epoch}: model accuracy {acc * 100} loss:\t{loss_test}")
        print(f"*" * 50)
    print(f"Best acc:{max(results['test_acc'])}")
    pd.DataFrame(data=results).to_csv(f"results_{MAX_EPOCHS}.cvs")





if __name__=="__main__":
    train_method(data_dir=os.path.dirname(os.getcwd()))
