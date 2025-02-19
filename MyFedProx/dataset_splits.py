import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from MyFedProx.SIIM_ISIC_Dataset import SIIM_ISIC_Dataset
import pandas as pd
from copy import deepcopy

"""
This file contains methods to download and split the SIIM_ISIC Dataset into pytorch dataloaders
"""


def fair_split(dataset: Dataset, nb_nodes: int, n_samples_per_node: int, batch_size: int, shuffle: bool) -> [DataLoader]:
    dataset_split = torch.utils.data.random_split(dataset, [1/nb_nodes for _ in range(nb_nodes)])
    dataloaders = [DataLoader(d, batch_size=batch_size, shuffle=shuffle) for d in dataset_split]
    return dataloaders

def unfair_size_split(dataset: Dataset, nb_nodes: int, batch_size, shuffle: bool) -> [DataLoader]:
    n_samples_per_node = []
    avg = 1/nb_nodes

    for i in range(nb_nodes if nb_nodes%2==0 else nb_nodes-1):
        n_samples_per_node.append(avg - avg/2 if i%2==0 else avg + avg/2)

    if nb_nodes%2!=0:
        n_samples_per_node.append(avg)
    dataset_split = torch.utils.data.random_split(dataset, n_samples_per_node)

    dataloaders = [DataLoader(d, batch_size=batch_size, shuffle=shuffle) for d in dataset_split]
    return dataloaders

    

def get_SIIM_ISIC(root_path, csv_path, type="normal", train_size=0.8, test_size=0.2, n_clients=3, batch_size=25, shuffle=True, device="cpu", total_size=None, resnet50=False, balanced=False):
    """
    Get the SIIM_ISIC dataset split into train and test dataloaders
    Args:
        - root_path: path to the folder containing the images
        - csv_path: path to the csv file containing the dataframe
        - type: "fair" or "non-fair"
        - train_size: size of the train dataset (between 0 and 1)
        - test_size: size of the test dataset (between 0 and 1)
        - n_clients: number of clients
        - batch_size: batch size
        - shuffle: shuffle the dataset
        - device: device to use for the dataset (cpu or cuda)
        - total_size: total size of the dataset (if None, then the whole dataset is used)
    """
    assert(type=="normal" or type=="fair" or type=="unfair_size")
    assert(train_size+test_size==1)

    # load the dataset contains in the folder root_path
    dataframe = pd.read_csv(csv_path)
    dataset = SIIM_ISIC_Dataset(root_path=root_path, dataframe=dataframe, device=device, total_size=total_size, resnet50=resnet50, balanced=balanced)

    # split the dataset into train and test
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    print(train_dataset, test_dataset)
    # make the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    
    if type=="fair":
        return  fair_split(train_dataset, n_clients, len(train_dataset)//n_clients, batch_size, shuffle), [deepcopy(test_dataloader) for _ in range(n_clients)]
    elif type=="unfair_size":
        return unfair_size_split(train_dataset, n_clients, batch_size, shuffle), [deepcopy(test_dataloader) for _ in range(n_clients)]

    return train_dataloader, test_dataloader

def plot_samples(data, title=None, plot_name="", n_examples =20):
    """
    Plot the first n_examples of the dataset
    Args:
        - data: the dataset
        - title: title of the plot
        - plot_name: name of the plot for saving
        - n_examples: number of examples to plot
    """

    n_rows = int(n_examples / 2)
    plt.figure(figsize=(128, 128))
    if title: plt.suptitle(title)
    X, y= data
    for idx in range(n_examples):

        ax = plt.subplot(n_rows, 2, idx + 1)

        image = X[idx].permute(1, 2, 0)
        if image.is_cuda: image = image.cpu()
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(str(y[idx].item()))

    if plot_name!="":plt.savefig(f"plots/"+plot_name+".png")

    plt.tight_layout()