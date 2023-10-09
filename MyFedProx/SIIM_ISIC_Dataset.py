import torch
import torchvision
from PIL import Image
from torchvision import transforms


class SIIM_ISIC_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, root_path, device="cpu", total_size=None, resnet50=False):
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
        
