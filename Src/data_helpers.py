import random
from os import listdir
from os.path import isfile, join
import numpy as np
import torch

from torch.utils.data import Dataset, Subset, DataLoader
import torchvision.transforms as transforms
from sklearn.utils.class_weight import  compute_class_weight
from sklearn.model_selection import train_test_split




def channel_rate_parsing(data_array: np.array):
    """
    Parse channel data from lodaed array. Take only title
    :param data_array:
    :return: vektor hodnot a label
    """
    title = data_array[0][1]  # take title
    data = np.array([float(rate) for channel, count, rate in data_array[6:]])  # take rate
    return data, title


def load_labeled_data(paths: dict, parse_file_function, maxdata_count=None):
    """
    Loads data from specified directories
    :param paths: dictionary key is directory with spectra, value is label.
    :param parse_file_function: function to parse data to array for neural net
    :param maxdata_count: maximum loaded instances (for testing purposes)
    :return: matrix with spectra, vector of labels, list of titles for each spectra
    """
    titles = []
    X = []
    y = []
    for path, label in paths.items():  # read all specified directories
        # print(f">>>{path}")
        for index, file in enumerate(listdir(path),1):  # for each file in directory
            f_path = join(path, file)
            if isfile(f_path):
                x_raw = np.load(f_path, allow_pickle=True)  # read raw data
                data, title = parse_file_function(x_raw)  # parse data, extract relevant information
                y.append(label)
                X.append(data)
                titles.append(title)
            if maxdata_count is not None and index == maxdata_count:
                break
    return np.vstack(X), np.array(y), titles


class SpectrumDataset(Dataset):
    """
    Dataset of measured signals
    """
    def __init__(self, paths_dict, parse_file_function=channel_rate_parsing, transform=None, maxdata_count=None):
        """
        If no transformation is given min-max scaling is used.
        """
        X, y, titles = load_labeled_data(paths_dict, parse_file_function=parse_file_function, maxdata_count=maxdata_count)
        self.transform = transform
        self.w = compute_class_weight("balanced",y=y,classes=[ i for i in paths_dict.values()]) # weights of classes
        self.X = torch.from_numpy(X).unsqueeze(1) # input data, ie. data on channels, we add one extra dimension, channel dimension=[samples,1,1024]
        if self.transform is not None:
            self.X = self.transform(self.X)
        else:
            x_min, _ = torch.min(self.X, dim=0 )
            x_max, _ = torch.max(self.X, dim=0)
            # print(x_min.shape)
            self.X = (self.X - x_min) / (x_max - x_min)

        self.labels = torch.from_numpy(y) # id of label must be coded into sequences of [0,0,..,0,1,0,..0]
        self._titles = titles # filename where the given signal is stored
       # self.one_hot_coder = torch.eye(n=len(paths_dict)) # generate identity matrix of shape number of classes
        print(f"Loaded dataset contains:{len(self.labels)}")
        freq = [ 0 for i in paths_dict.values()]
        for label in self.labels:
            freq[label]+=1
        for i in range(len(freq)):
            print(f"Label:{i}\tfrequency:{freq[i]}\tweights:{self.w[i]}")
        self.w = torch.from_numpy(self.w).float() # weights are used in model there must be set to float


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.X[idx],  self.labels[idx]   #self.one_hot_coder[self.labels[idx]]
    @property
    def weights(self):
        """
        Weights for classes
        :return:
        """
        return self.w
    @property
    def titles(self):
        """
        File name for each data sample, i.e. filename where the data are stored
        :return:
        """
        return self._titles

    @property
    def X_data(self):
        return self.X

    @property
    def y_data(self):
        return self.labels

class RotatedSpectrumDataset(Dataset):
    def __init__(self, traindataset, left=-5, right=5, mult=5):
        self.X = []
        self.labels=[]
        for x, y in traindataset:
            self.X.append(x)
            self.labels.append(y)
            for i in range(mult):
                rot = random.randint(left, right)
                signal = torch.roll(x, shifts=rot, dims=1)
                self.X.append(signal)
                self.labels.append(y)
        self.X=torch.stack(self.X)
        self.labels=torch.stack(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.X[idx], self.labels[idx]  # self.one_hot_coder[self.labels[idx]]


def train_test_dataset_split(dataset: SpectrumDataset, test_split=0.2, seed=42):
    """
    Split dataset into training and testing subsets.
    :param dataset:
    :param test_split:
    :param seed:
    :return:
    """
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=test_split, random_state=seed)
    return Subset(dataset, train_idx), Subset(dataset, test_idx)



def roll_batched_channels(signals, left=-2, right=2):
    """
    Rotate batch of signals
    :param signals: 
    :param left: 
    :param right: 
    :return: rotated signal
    """
    rot=random.randint(left, right)
    signals=torch.roll(signals, shifts=rot,dims=2)
    return signals



def channel_statistics(dataset: SpectrumDataset, num_channels):
    """
    Compute statistics mu and std over dataset.
    :param dataset:
    :return:
    """
    channel_sum = torch.zeros((1))
    channel_sum_diff = torch.zeros((1))
    for channels_data, label in dataset:
        channel_sum += torch.sum(channels_data,dim=(1))
    channel_mean = channel_sum / (len(dataset) * num_channels)
    for channels_data, label in dataset:
        channel_sum_diff += (torch.sum(channels_data, dim=(1)) / num_channels - channel_mean) ** 2
    channel_std = torch.sqrt(channel_sum_diff / len(dataset))
    print(f"Data set mean: {channel_mean}")
    print(f"Data set std: {channel_std}")
    return channel_mean, channel_std

if __name__ == "__main__":

    #X, y, titles = load_labeled_data({"../Data/zdroj_co/": 0}, parse_file_function=channel_rate_parsing)
    label_dict = {'../Data/zdroj_co': 0,
                  '../Data/zdroj_cs': 1,
                  '../Data/zdroj_eu_152': 2,
                  '../Data/zdroj_th_232': 3,
                  '../Data/zdroj_u_238': 4,
                  '../Data/zdroj_am_241': 5,
                  '../Data/zdroj_co_cs': 6
                  }
    # Before normalization
    # Data set mean: tensor([4.8696])
    # Data set std: tensor([3.1731])
    sdata = SpectrumDataset(paths_dict=label_dict, transform=transforms.Normalize((4.8696), (3.1731)))
    channel_statistics(sdata, num_channels=1024)


    train_dataloader = DataLoader(sdata, batch_size=2, shuffle=False)
    for bid, batch in enumerate(train_dataloader):
        x,y = batch
        roll_batched_channels(x)
        exit(0)

