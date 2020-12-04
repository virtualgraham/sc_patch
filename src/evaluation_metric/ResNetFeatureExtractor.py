import torch
import torch.nn as nn

import torchvision
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

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


class ResNetFeatureExtractor():
    def __init__(self):)
        model = torchvision.models.resnet50(pretrained=True)).to(device)

        modules=list(model.children())[:-1]
        model=nn.Sequential(*modules)

        for p in model.parameters():
            p.requires_grad = False

        model.eval()

        self.model = model
        
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