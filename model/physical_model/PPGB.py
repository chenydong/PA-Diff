import torch
from torch import nn
import numpy as np
from torchvision.transforms import ToTensor
from PIL import ImageFilter
from .ConditionNet import ConditionNet
from ..utils import np_to_pil, torch_to_np
from torchvision.transforms import ToTensor

def get_A(x):
    # x_np = np.clip(torch_to_np(x), 0, 1)
    # x_pil = np_to_pil(x_np)
    # h, w = x_pil.size
    # windows = (h + w) / 2
    # A = x_pil.filter(ImageFilter.GaussianBlur(windows))
    # A = ToTensor()(A).to(x.device)
    # return A.unsqueeze(0)

    processed_images = []

    for i in range(x.size(0)):
        img_np = torch_to_np(x[i])
        img_pil = np_to_pil(img_np)
        
        h, w = img_pil.size
        windows = (h + w) / 2
        
        A_pil = img_pil.filter(ImageFilter.GaussianBlur(windows))
        
        A_tensor = ToTensor()(A_pil).to(x.device)
        processed_images.append(A_tensor)
    
    return torch.stack(processed_images, dim=0)

class TNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.aNet = ConditionNet(support_size=input_channels)
        self.final = nn.Conv2d(128, output_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        a = self.final(self.aNet(x))
        return a
 
class PM(nn.Module):
    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.tNet = TNet(input_channels,1)
        self.aNet = TNet(input_channels,3)

    def forward(self, x): 
        a = self.aNet(get_A(x)+x)
        t = self.tNet(x)
        return a, t
    