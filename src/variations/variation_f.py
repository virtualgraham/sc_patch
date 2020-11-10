import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, datasets
 
import torchvision
from torchvision import transforms
from torchvision import models
 
import torch.nn.functional as F
import torchvision.transforms.functional as TF
 
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import time

from tqdm import tqdm # progress bar

import skimage
import cv2

from glob import glob

from torchsummary import summary

import math
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Rotation, no jigsaw
# Saliency Check
# No color shift or grayscale conversion

#########################################
# Parameters 
#########################################

training_image_paths = glob('/data/open-images-dataset/train/*.jpg')
validation_image_paths = glob('/data/open-images-dataset/validation/*.jpg')

train_dataset_length = 409600
validation_dataset_length = 20480
train_batch_size = 1024
validation_batch_size = 1024
num_epochs = 1500
save_after_epochs = 1 
backup_after_epochs = 5 
model_save_prefix = "variation_f"
reuse_image_count = 4

patch_dim = 64

learn_rate = 0.0001
momentum = 0.974
weight_decay = 0.0005



#########################################
# Utilities 
#########################################

def imshow(img,text=None,should_save=False):
    plt.figure(figsize=(10, 10))
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()  

def show_plot(iteration,loss,fname):
    plt.plot(iteration,loss)
    plt.savefig(fname)
    plt.show()
    
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for i, t in enumerate(tensor):
            t.mul_(self.std[i%3]).add_(self.mean[i%3])
        return tensor

unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))



#########################################
# This class generates patches for training
#########################################

class ShufflePatchDataset(Dataset):

  def __init__(self, image_paths, patch_dim, length, transform=None):
    self.image_paths = image_paths
    self.patch_dim = patch_dim
    self.length = length
    self.transform = transform
    self.image_reused = 0
    
    self.margin = 10
    self.window_width = self.patch_dim + 2*self.margin

    self.min_image_width = self.window_width + 1

    self.saliency = cv2.saliency.StaticSaliencyFineGrained_create()

  def __len__(self):
    return self.length
  

  def saliency_check(self, window):
    (success, saliency_map) = self.saliency.computeSaliency(cv2.cvtColor(window, cv2.COLOR_RGB2BGR))

    patch_saliency_map = saliency_map[self.margin:self.margin+self.patch_dim, self.margin:self.margin+self.patch_dim]
    patch_saliency = np.sum(patch_saliency_map > .5)
    print('patch_saliency', patch_saliency)
    
    return patch_saliency >= 180 # 100 + 40 + 40


  def __getitem__(self, index):
    # [y, x, chan], dtype=uint8, top_left is (0,0)
    
    image_index = int(math.floor((len(self.image_paths) * random.random())))
    
    if self.image_reused == 0:
      pil_image = Image.open(self.image_paths[image_index]).convert('RGB')
      self.pil_image = pil_image.resize((int(round(pil_image.size[0]/3)), int(round(pil_image.size[1]/3))))
      self.image_reused = reuse_image_count
    else:
      self.image_reused -= 1

    image = np.array(self.pil_image)

    # If image is too small, try another image
    if (image.shape[0] - self.min_image_width) <= 0 or (image.shape[1] - self.min_image_width) <= 0:
        return self.__getitem__(index)
    
    window_y_coord = int(math.floor((image.shape[0] - self.window_width) * random.random()))
    window_x_coord = int(math.floor((image.shape[1] - self.window_width) * random.random()))

    window = image[window_y_coord:window_y_coord+self.window_width, window_x_coord:window_x_coord+self.window_width]
    
    if not self.saliency_check(window):
      return self.__getitem__(index)

    rotation_label = int(math.floor((4 * random.random())))

    if rotation_label>0:
      window = np.rot90(window, rotation_label).copy()

    patch = window[self.margin:self.margin+self.patch_dim, self.margin:self.margin+self.patch_dim]
        
    if self.transform:
      patch = self.transform(patch)

    return patch, np.array(rotation_label).astype(np.int64)
    

##################################################
# Creating Train/Validation dataset and dataloader
##################################################

traindataset = ShufflePatchDataset(training_image_paths, patch_dim, train_dataset_length
                         transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

trainloader = torch.utils.data.DataLoader(traindataset, 
                                          batch_size=train_batch_size,
                                          num_workers=4,
                                          shuffle=False)


valdataset = ShufflePatchDataset(validation_image_paths, patch_dim, validation_dataset_length
                         transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

valloader = torch.utils.data.DataLoader(valdataset,
                                        batch_size=validation_batch_size,
                                        num_workers=4,
                                        shuffle=False)



##################################################
# Model for learning patch position
##################################################

class VggNetwork(nn.Module):
  def __init__(self,aux_logits = False):

      super(VggNetwork, self).__init__()

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

        nn.MaxPool2d(kernel_size=2, stride=2),


        nn.Conv2d(256, 512, kernel_size=3, padding=1),
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

        nn.MaxPool2d(kernel_size=2, stride=2)
        
      )
    
      self.fc6 = nn.Sequential(
        nn.Linear(512 * 2 * 2, 4096),
        nn.ReLU(True),
        nn.Dropout(),
      )

      self.fc = nn.Sequential(
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 24),
      )

  def forward_once(self, x):
    output= self.cnn(x)
    output = output.view(output.size()[0], -1)
    output = self.fc6(output)
    return output

  def forward(self, patch):
    output_fc6_patch = self.forward_once(patch)
    output = self.fc(output_fc6_patch)
    return output, output_fc6_patch

model = VggNetwork().to(device)
summary(model, [(3, patch_dim, patch_dim)])



#############################################
# Initialized Optimizer, criterion, scheduler
#############################################

optimizer = optim.SGD(
  model.parameters(), 
  lr=learn_rate,
  momentum=momentum,
  weight_decay=weight_decay
)

criterion = nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                           mode='min',
                                           patience=5,
                                           factor=0.5, verbose=True)

#############################################
# Load Checkpoint
#############################################

global_trn_loss = []
global_val_loss = []

last_epoch = -1

training_image_paths = glob(f'{model_save_prefix}_{train_batch_size}_{num_epochs}_{learn_rate}_{patch_dim}_{gap}_*.pt')

if len(training_image_paths) > 0:
  training_image_paths.sort()  
  model_save_path = training_image_paths[-1]
  try:
    print('Loading Checkpoint...', model_save_path)
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    last_epoch = checkpoint['epoch']
    scheduler = checkpoint['scheduler']
    loss = checkpoint['loss']
    global_trn_loss = checkpoint['global_trnloss']
    global_val_loss = checkpoint['global_valloss']
  except:
    print("Loading Checkpoint Failed")



############################
# Training/Validation Engine
############################

print("starting train loop (b.0)")

for epoch in range(last_epoch+1, num_epochs):
    print("epoch", epoch)

    train_running_loss = []
    val_running_loss = []
    start_time = time.time()
    model.train()
    for idx, data in tqdm(enumerate(trainloader), total=int(len(traindataset)/train_batch_size)):
        patch, patch_shuffle_order_label = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output, output_fc6_patch = model(patch)
        loss = criterion(output, patch_shuffle_order_label)
        loss.backward()
        optimizer.step()
        
        train_running_loss.append(loss.item())
    else:
      correct = 0
      total = 0
      model.eval()
      with torch.no_grad():
        for idx, data in tqdm(enumerate(valloader), total=int(len(valdataset)/validation_batch_size)):
          patch, patch_shuffle_order_label = data[0].to(device), data[1].to(device)
          output, output_fc6_patch = model(patch)
          loss = criterion(output, patch_shuffle_order_label)
          val_running_loss.append(loss.item())
        
          _, predicted = torch.max(output.data, 1)
          total += patch_shuffle_order_label.size(0)
          correct += (predicted == patch_shuffle_order_label).sum()
        print('Val Progress --- total:{}, correct:{}'.format(total, correct.item()))
        print('Val Accuracy of the network on the test images: {}%'.format(100 * correct.item() / total))

    global_trn_loss.append(sum(train_running_loss) / len(train_running_loss))
    global_val_loss.append(sum(val_running_loss) / len(val_running_loss))

    scheduler.step(global_val_loss[-1])

    print('Epoch [{}/{}], TRNLoss:{:.4f}, VALLoss:{:.4f}, Time:{:.2f}'.format(
        epoch + 1, num_epochs, global_trn_loss[-1], global_val_loss[-1],
        (time.time() - start_time) / 60))
    
    if epoch % save_after_epochs == 0:

      # delete old images
      training_image_paths = glob(f'{model_save_prefix}_{train_batch_size}_{num_epochs}_{learn_rate}_{patch_dim}_{gap}_*.pt')
      if len(training_image_paths) > 2:
        training_image_paths.sort()
        for i in range(len(training_image_paths)-2):
          training_image_path = training_image_paths[i]
          os.remove(training_image_path)

      # save new image
      model_save_path = f'{model_save_prefix}_{train_batch_size}_{num_epochs}_{learn_rate}_{patch_dim}_{gap}_{epoch:04d}.pt'
      print('saving checkpoint', model_save_path)
      torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler': scheduler,
            'loss': loss,
            'global_trnloss': global_trn_loss,
            'global_valloss': global_val_loss
        }, model_save_path)
    
      if epoch % backup_after_epochs == 0:
        print('backing up checkpoint', model_save_path)
        os.system(f'aws s3 cp /data/{model_save_path} s3://guiuan/{model_save_prefix}_{epoch:04d}_{learn_rate}_{global_trn_loss[-1]:.4f}_{(100 * correct.item()/total):.2f}.pt')

print("done")