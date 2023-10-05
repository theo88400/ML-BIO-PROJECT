import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from MyFedProx.SIIM_ISIC_Dataset import SIIM_ISIC_Dataset
import pandas as pd

"""
This file contains methods to download and split the SIIM_ISIC Dataset into pytorch dataloaders
"""


def iid_split(dataset: Dataset, nb_nodes: int, n_samples_per_node: int, batch_size: int, shuffle: bool) -> [DataLoader]:
    # TODO: will be implemented later
    pass

def non_iid_split(dataset: Dataset, nb_nodes: int, n_samples_per_node: int, batch_size: int, shuffle: int, shuffle_digits=False) -> [DataLoader]:
    # TODO: will be implemented later
    pass

def get_SIIM_ISIC(root_path, csv_path, type="iid", train_size=0.8, test_size=0.2, n_clients=3, batch_size=25, shuffle=True, device="cpu", total_size=None):
    assert(type=="iid" or type=="non-iid")
    assert(train_size+test_size==1)

    # load the dataset contains in the folder root_path
    dataframe = pd.read_csv(csv_path)
    dataset = SIIM_ISIC_Dataset(root_path=root_path, dataframe=dataframe, device=device, total_size=total_size)

    # split the dataset into train and test
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # make the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return train_dataloader, test_dataloader

def plot_samples(data, title=None, plot_name="", n_examples =20):

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