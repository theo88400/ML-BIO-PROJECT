import torch
import torchvision
from PIL import Image
from torchvision import transforms
from sklearn.utils import resample
import pandas as pd


class SIIM_ISIC_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, root_path, device="cpu", total_size=None, resnet50=False, balanced=False):
        # dataframe contains info about patients as well as image path and label
        self.dataframe = dataframe
        self.root_path = root_path
        self.device = device
        self.total_size = total_size
        self.resnet50 = resnet50
        self.preprocess = transforms.Compose([
                            transforms.Resize(224),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
        
        if balanced and total_size is None:
            # downsample the majority class to have a 50/50 split
            df_min = self.dataframe[self.dataframe["target"] == 1]
            df_maj = self.dataframe[self.dataframe["target"] == 0]
            df_maj_downsampled = df_maj.sample(n=len(df_min))
            self.dataframe = pd.concat([df_min, df_maj_downsampled])
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)



    def __len__(self):
        return len(self.dataframe) if self.total_size is None else self.total_size

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        path = self.root_path + row["image_name"] + ".jpg"
        img = Image.open(path)
        if self.resnet50:
            tensor = self.preprocess(img)
        else:
            tensor = torchvision.transforms.functional.resize(torchvision.transforms.functional.to_tensor(img), (224, 224))
        return (
            tensor.to(self.device),
            torch.nn.functional.one_hot(torch.tensor(row["target"]), num_classes=2).to(self.device),
        )
        
