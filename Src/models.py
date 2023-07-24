import torch
import torch.nn as nn
from torch.nn import functional as F


class CNNConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size, dropout) -> None:
        super(CNNConvolutionBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_size))
        if dropout is not None:
            self.conv.add_module("dropout", nn.Dropout(dropout))


    def forward(self, x):
        return self.conv(x)


class DenseBlock(nn.Module):
    def __init__(self, input_size, output_size, activation, dropout) -> None:
        super(DenseBlock, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(input_size, output_size),
        )
        if activation:
            self.dense.add_module("relu", nn.ReLU())
        if dropout is not None:
            self.dense.add_module("dropout", nn.Dropout(dropout))

    def forward(self, x):
        return self.dense(x)


class CNN1Classifier(nn.Module):
    def __init__(self, cnn_config, dense_config):
        super(CNN1Classifier, self).__init__()

        self.cnn_modules = nn.ModuleList()
        for conf in cnn_config:
            self.cnn_modules.append(CNNConvolutionBlock(
                in_channels=conf["in_channels"],
                out_channels=conf["out_channels"],
                kernel_size=conf["kernel_size"],
                pool_size=conf["pool_size"])
            )
        self.flat = nn.Flatten()
        self.dense_modules = nn.ModuleList()
        for conf in dense_config:
            self.dense_modules.append(DenseBlock(
                input_size=conf["in"],
                output_size=conf["out"],
                activation=conf["activation"],
                dropout=conf["dropout"]))

    def forward(self, x):
        out = self.cnn_modules[0](x)
        for mod in self.cnn_modules[1:]:
            out = mod(out)
        out = self.flat(out)

        for mod in self.dense_modules:
            out = mod(out)

        return F.softmax(out, dim=1)

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)



