import torch
import torchvision
from PIL import Image


class SIIM_ISIC_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, root_path, device="cpu", total_size=None):
        # dataframe contains info about patients as well as image path and label
        self.dataframe = dataframe
        self.root_path = root_path
        self.device = device
        self.total_size = total_size

    def __len__(self):
        return len(self.dataframe) if self.total_size is None else self.total_size

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        path = self.root_path + row["image_name"] + ".jpg"
        return (
            torchvision.transforms.Resize((256, 256))(torchvision.transforms.functional.to_tensor(Image.open(path)).to(self.device)),
            torch.tensor(row["target"]).to(self.device),
        )
        
