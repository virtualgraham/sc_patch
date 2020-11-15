import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms

from glob import glob

import math
import random 

import numpy as np
from PIL import Image
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ShufflePatchEvalNet(nn.Module):
  def __init__(self,aux_logits = False):

      super(ShufflePatchEvalNet, self).__init__()

      self.cnn = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64), 
        nn.ReLU(inplace=True),

        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64), 
        nn.ReLU(inplace=True),

        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128), 
        nn.ReLU(inplace=True),

        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128), 
        nn.ReLU(inplace=True),

        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256), 
        nn.ReLU(inplace=True),

        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256), 
        nn.ReLU(inplace=True),

        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256), 
        nn.ReLU(inplace=True),

        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(256, 512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512), 
        nn.ReLU(inplace=True),

        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512), 
        nn.ReLU(inplace=True),

        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512), 
        nn.ReLU(inplace=True),

        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512), 
        nn.ReLU(inplace=True),

        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512), 
        nn.ReLU(inplace=True),

        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512), 
        nn.ReLU(inplace=True),

        nn.MaxPool2d(kernel_size=2, stride=2)
      )
    
  def forward(self, patch):
    return self.cnn(patch)



class ToTensor:
    def __init__(self):
        self.max = 255
        
    def __call__(self, tensor):
        return tensor.float().div_(self.max)


class Normalize:
    def __init__(self, mean, std, inplace=False, dtype=torch.float, device='cpu'):
        self.mean = torch.as_tensor(mean, dtype=dtype, device=device)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype, device=device)[None, :, None, None]
        self.inplace = inplace
        
    def __call__(self, tensor):
        if not self.inplace:
            tensor = tensor.clone()

        tensor.sub_(self.mean).div_(self.std)
        return tensor

    
class ShufflePatchFeatureExtractor():
    def __init__(self, weights_path):
        self.model = ShufflePatchEvalNet().to(device)
        
        print('Loading Weights...', weights_path)
        checkpoint = torch.load(weights_path, map_location=torch.device(device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        
        self.transform_batch = transforms.Compose([
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        )
               
    # Numpy array of size (N, H, W, C)
    # Used for PIL images
    def evalRGB(self, patches):
        patches = torch.from_numpy(patches)
        patches = patches.permute(0, 3, 1, 2)
        patches = self.transform_batch(patches)
        output = self.model(patches.to(device))
        return output.cpu().detach().numpy()

    # Numpy array of size (N, H, W, C)
    # Used for CV2 images
    def evalBGR(self, patches):
        patches = patches[...,::-1].copy()
        return self.evalRGB(patches)