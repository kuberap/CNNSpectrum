import os
import json
from functools import partial

import torch
from torch.utils.data import Dataset, Subset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler


from optim_7_classes_config import *
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



def create_net_config(config):
    cnn_conf = [
        {"in_channels": 1,
         "out_channels": config["cnn1-out"],
         "kernel_size": config["cnn1-kernel"],
         "pool_size": 2,  # division by 2 is universal
         "dropout": config["cnn1-dropout"]
         },
        {"in_channels": config["cnn1-out"],  # must be the same as the output from the layer one
         "out_channels": config["cnn2-out"],
         "kernel_size": config["cnn2-kernel"],
         "pool_size": 2,  # division by 2 is universal
         "dropout": config["cnn2-dropout"]
         }
    ]
    flatten_size = compute_flatten_size(cnn_conf)
    dense_conf = [
        {"in": flatten_size,
         "out": config["dense1_out"],
         "activation": True,
         "dropout": config["dense1-dropout"]},
        {
            "in": config["dense1_out"],# must be the same as the output from the layer one
            "out": len(LABEL_DICT),
            "activation": False,
            "dropout": None}
    ]

    return cnn_conf, dense_conf

def train_method(config, data_dir=None):
    # Before normalization:  Data set mean: tensor([4.8696]) Data set std: tensor([3.1731])

    # special hack
    remote_label_dict={data_dir+k:v for k,v in LABEL_DICT.items()}

    spectredataset = SpectrumDataset(paths_dict=remote_label_dict, transform=transforms.Normalize((4.8696), (3.1731)))
    train_dataset, test_dataset = train_test_dataset_split(spectredataset)
    # train_dataset, val_dataset = train_test_dataset_split(train_dataset, test_split=config["val_split"])

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True)

    cnn_conf, dense_conf = create_net_config(config)
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


    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])  # lr=0.0001

    checkpoint = session.get_checkpoint()
    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        model.load_state_dict(checkpoint_state["model_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    print("Start - learning")

    for epoch in range(0, config["max_epochs"]):
        loss_train = 0.0
        acc = 0
        count = 0
        train_epoch_steps=0
        for bid, batch in enumerate(train_dataloader):
            inputs, labels = batch
            if torch.rand(1) < config["p-rotation"]:
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
            #print(f"\tEPOCH:{epoch}:{bid}--->{item}::::{loss_train}")
            acc += (torch.argmax(y_pred, 1) == labels).float().sum()
            count += len(labels)
            train_epoch_steps+=1
        acc /= count
        loss_train/=train_epoch_steps
        # print(f"Train>Epoch {epoch}/{config['max_epochs']}: model accuracy {acc * 100} loss:\t{loss_train}")

        acc = 0
        count = 0
        loss_vall = 0.0
        val_epoch_steps = 0
        for inputs, labels in val_dataloader:
            with torch.no_grad():
                inputs = inputs.to(device=device, dtype=torch.float16)
                labels = labels.to(device=device)
                with torch.autocast(device_type=device.type, enabled=True):
                    y_pred = model(inputs)
                    loss = loss_fn(y_pred, labels)

                item = loss.item()
                loss_vall += item
                acc += (torch.argmax(y_pred, 1) == labels).float().sum()
                count += len(labels)
                val_epoch_steps+=1

        acc /= count
        loss_vall/=val_epoch_steps
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)

        session.report(
            {"loss": loss_vall, "accuracy": acc.item()},
            checkpoint=checkpoint,
        )
        # print(f"Vall>Epoch {epoch}: model accuracy {acc * 100} loss:\t{loss_vall}")
        # print(f"*" * 50)





def optimize():

    config={
        "cnn1-out":tune.choice([8,16,32,64,128]),
        "cnn1-kernel":tune.choice([3,5]),
        "cnn1-dropout":tune.choice([0.1,0.05,0.01,0.001]),
        "cnn2-out":tune.choice([8,16,32,64,128]),
        "cnn2-kernel":tune.choice([3,5]),
        "cnn2-dropout": tune.choice([0.1,0.05,0.01,0.001]),
        "dense1_out": tune.choice([2 ** i for i in range(4,11)]),
        "dense1-dropout": tune.choice([0.1,0.05,0.01,0.001]),
        "batch_size":tune.choice([32,64,128,256,512]),
        "lr":tune.loguniform(1e-4, 1e-1),
        "momentum":0.95,
        "p-rotation":tune.choice([0,0.1,0.2, 0.3]),
        "val_split":0.1,
        "max_epochs":MAX_OPTIM_EPOCHS

    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=MAX_OPTIM_EPOCHS,
        grace_period=1,
        reduction_factor=2,
    )
    result = tune.run(
        partial(train_method, data_dir=os.path.dirname(os.getcwd())),
        resources_per_trial={"cpu": 32, "gpu": 2},
        config=config,
        num_samples=OPT_MAX_SAMPLES,
        scheduler=scheduler,
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")
    # train_method(config)
    with open('../TmpRes/best_conf-2cnn_1dense.txt', 'w') as convert_file:
        convert_file.write(json.dumps(best_trial.config))






if __name__=="__main__":
    optimize()







