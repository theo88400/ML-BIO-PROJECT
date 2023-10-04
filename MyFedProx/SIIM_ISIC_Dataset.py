import torch
import torchvision
from PIL import Image


class SIIM_ISIC_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, root_path):
        # dataframe contains info about patients as well as image path and label
        self.dataframe = dataframe
        self.root_path = root_path

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        path = self.root_path + row["image_name"] + ".jpg"
        return (
            torchvision.transforms.Resize((256, 256))(torchvision.transforms.functional.to_tensor(Image.open(path))),
            row["target"],
        )
        
