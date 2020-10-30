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
from skimage import img_as_ubyte, img_as_float32

from sklearn.model_selection import StratifiedShuffleSplit

from glob import glob

from torchsummary import summary

import math
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)



#########################################
# Parameters 
#########################################

training_image_paths = glob('Objects365/train/*.jpg')
validation_image_paths = glob('Objects365/val/*.jpg')
patch_dim = 96
train_dataset_length = 40192 # 40192
validation_dataset_length = 2048 # 2048
gap = 48
jitter = 7
train_batch_size = 256
validation_batch_size = 128
num_epochs = 1500

learn_rate = 0.0000625
momentum = 0.974
weight_decay = 0.0005

save_after_epochs = 1 



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

class MyDataset(Dataset):

  def __init__(self, image_paths, patch_dim, length, gap, jitter, transform=None):
    self.image_paths = image_paths
    self.patch_dim = patch_dim
    self.length = length
    self.gap = gap
    self.jitter = jitter
    self.transform = transform

  def __len__(self):
    return self.length
  
  def prep_patch(self, image):
    # print('prep_patch image.shape', image.shape)
    # for some patches, randomly downsample to as little as 100 total pixels
    if(random.random() < .33):
      pil_patch = Image.fromarray(image)
      original_size = pil_patch.size
      randpix = int(math.sqrt(random.random() * (95 * 95 - 10 * 10) + 10 * 10))
      pil_patch = pil_patch.resize((randpix, randpix)) 
      pil_patch = pil_patch.resize(original_size) 
      np.copyto(image, np.array(pil_patch))

    # randomly drop all but one color channel
    chan_to_keep = random.randint(0, 2)
    for i in range(0, 3):
      if i != chan_to_keep:
        image[:,:,i] = np.random.randint(0, 255, (self.patch_dim, self.patch_dim), dtype=np.uint8)


  def __getitem__(self, index):
    # [y, x, chan], dtype=uint8, top_left is (0,0)
        
    patch_loc_arr = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    
    image_index = int(math.floor((len(self.image_paths) * random.random())))
    
    pil_image = Image.open(self.image_paths[image_index]).convert('RGB')

    # Imagenet 150000 -> 180000 -> 450000
    # 0.826 -> 2.479
    # Objects365 300000 -> 370000 -> 925000

    # original_size = pil_image.size
    # randpix = int(math.sqrt(random.random() * (95 * 95 - 10 * 10) + 10 * 10))
    # pil_image = pil_image.resize((randpix, randpix)) 

    image = np.array(pil_image)

    # If image is too small, try another image
    min_width = 3*patch_dim + 2*jitter + 2*gap
    if (image.shape[0] - min_width) <= 0 or (image.shape[1] - min_width) <= 0:
        return self.__getitem__(index)
    
    margin = math.ceil(patch_dim/2.0) + jitter

    patch_direction_label = int(math.floor((8 * random.random())))
    
    patch_jitter_y = int(math.floor((self.jitter * 2 * random.random()))) - self.jitter
    patch_jitter_x = int(math.floor((self.jitter * 2 * random.random()))) - self.jitter
        
    while True:
        
        uniform_patch_y_coord = int(math.floor((image.shape[0] - margin*2) * random.random())) + margin - int(round(self.patch_dim/2.0))
        uniform_patch_x_coord = int(math.floor((image.shape[1] - margin*2) * random.random())) + margin - int(round(self.patch_dim/2.0))

        random_patch_y_coord = uniform_patch_y_coord + patch_loc_arr[patch_direction_label][0] * (self.patch_dim + self.gap) + patch_jitter_y
        random_patch_x_coord = uniform_patch_x_coord + patch_loc_arr[patch_direction_label][1] * (self.patch_dim + self.gap) + patch_jitter_x

        if random_patch_y_coord >= 0 and random_patch_x_coord >= 0 and random_patch_y_coord < (image.shape[0] - patch_dim) and random_patch_x_coord < (image.shape[1] - patch_dim):
            break

    uniform_patch = image[uniform_patch_y_coord:uniform_patch_y_coord+self.patch_dim, uniform_patch_x_coord:uniform_patch_x_coord+self.patch_dim]        
    random_patch = image[random_patch_y_coord:random_patch_y_coord+self.patch_dim, random_patch_x_coord:random_patch_x_coord+self.patch_dim]
        
    self.prep_patch(uniform_patch)
    self.prep_patch(random_patch)

    patch_direction_label = np.array(patch_direction_label).astype(np.int64)
        
    if self.transform:
      uniform_patch = self.transform(uniform_patch)
      random_patch = self.transform(random_patch)

    return uniform_patch, random_patch, patch_direction_label



##################################################
# Creating Train/Validation dataset and dataloader
##################################################

traindataset = MyDataset(training_image_paths, patch_dim, train_dataset_length, gap, jitter,
                         transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

trainloader = torch.utils.data.DataLoader(traindataset, 
                                          batch_size=train_batch_size,
                                          shuffle=False)


valdataset = MyDataset(validation_image_paths, patch_dim, validation_dataset_length, gap, jitter,
                         transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

valloader = torch.utils.data.DataLoader(valdataset,
                                        batch_size=validation_batch_size,
                                        shuffle=False)



##################################################
# Model for learning patch position
##################################################

class AlexNetwork(nn.Module):
  def __init__(self,aux_logits = False):

      super(AlexNetwork, self).__init__()

      self.cnn = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        # nn.BatchNorm2d(64), 
        nn.ReLU(inplace=True),

        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        # nn.BatchNorm2d(64), 
        nn.ReLU(inplace=True),

        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        # nn.BatchNorm2d(128), 
        nn.ReLU(inplace=True),

        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        # nn.BatchNorm2d(128), 
        nn.ReLU(inplace=True),

        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        # nn.BatchNorm2d(256), 
        nn.ReLU(inplace=True),

        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        # nn.BatchNorm2d(256), 
        nn.ReLU(inplace=True),

        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        # nn.BatchNorm2d(256), 
        nn.ReLU(inplace=True),

        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(256, 512, kernel_size=3, padding=1),
        # nn.BatchNorm2d(512), 
        nn.ReLU(inplace=True),

        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        # nn.BatchNorm2d(512), 
        nn.ReLU(inplace=True),

        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        # nn.BatchNorm2d(512), 
        nn.ReLU(inplace=True),

        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        # nn.BatchNorm2d(512), 
        nn.ReLU(inplace=True),

        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        # nn.BatchNorm2d(512), 
        nn.ReLU(inplace=True),

        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        # nn.BatchNorm2d(512), 
        nn.ReLU(inplace=True),

        nn.MaxPool2d(kernel_size=2, stride=2)
      )
    
      self.fc6 = nn.Sequential(
        nn.Linear(512 * 3 * 3, 4096),
        nn.ReLU(True),
        nn.Dropout(),
      )

      self.fc = nn.Sequential(
        nn.Linear(2*4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 8),
      )

  def forward_once(self, x):
    output= self.cnn(x)
    #print('a', output.size())
    output = output.view(output.size()[0], -1)
    #print('b', output.size())
    output = self.fc6(output)
    return output

  def forward(self, uniform_patch, random_patch):
    output_fc6_uniform = self.forward_once(uniform_patch)
    output_fc6_random = self.forward_once(random_patch)
    output = torch.cat((output_fc6_uniform,output_fc6_random), 1)
    output = self.fc(output)
    return output, output_fc6_uniform, output_fc6_random

model = AlexNetwork().to(device)
summary(model, [(3, 96, 96), (3, 96, 96)])



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

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
#                                            mode='min',
#                                            patience=10,
#                                            factor=0.3, verbose=True)



#############################################
# Load Checkpoint
#############################################

global_trn_loss = []
global_val_loss = []

last_epoch = -1

training_image_paths = glob(f'model_b_{train_batch_size}_{num_epochs}_{learn_rate}_{patch_dim}_{gap}_*.pt')

if len(training_image_paths) > 0:
  training_image_paths.sort()  
  model_save_path = training_image_paths[-1]
  try:
    print('Loading Checkpoint...', model_save_path)
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    last_epoch = checkpoint['epoch']
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
        uniform_patch, random_patch, random_patch_label = data[0].to(device), data[1].to(device), data[2].to(device)
        # print(uniform_patch.size(), random_patch.size())
        optimizer.zero_grad()
        output, output_fc6_uniform, output_fc6_random = model(uniform_patch, random_patch)
        loss = criterion(output, random_patch_label)
        loss.backward()
        optimizer.step()
        
        train_running_loss.append(loss.item())
    else:
      correct = 0
      total = 0
      model.eval()
      with torch.no_grad():
        for idx, data in tqdm(enumerate(valloader), total=int(len(valdataset)/validation_batch_size)):
          uniform_patch, random_patch, random_patch_label = data[0].to(device), data[1].to(device), data[2].to(device)
          output, output_fc6_uniform, output_fc6_random = model(uniform_patch, random_patch)
          loss = criterion(output, random_patch_label)
          val_running_loss.append(loss.item())
        
          _, predicted = torch.max(output.data, 1)
          total += random_patch_label.size(0)
          correct += (predicted == random_patch_label).sum()
        print('Val Progress --- total:{}, correct:{}'.format(total, correct.item()))
        print('Val Accuracy of the network on the test images: {}%'.format(100 * correct.item() / total))

    global_trn_loss.append(sum(train_running_loss) / len(train_running_loss))
    global_val_loss.append(sum(val_running_loss) / len(val_running_loss))

    # scheduler.step(global_val_loss[-1])

    print('Epoch [{}/{}], TRNLoss:{:.4f}, VALLoss:{:.4f}, Time:{:.2f}'.format(
        epoch + 1, num_epochs, global_trn_loss[-1], global_val_loss[-1],
        (time.time() - start_time) / 60))
    
    if epoch % save_after_epochs == 0:

      # delete old images
      training_image_paths = glob(f'model_b_{train_batch_size}_{num_epochs}_{learn_rate}_{patch_dim}_{gap}_*.pt')
      if len(training_image_paths) > 2:
        training_image_paths.sort()
        for i in range(len(training_image_paths)-2):
          training_image_path = training_image_paths[i]
          os.remove(training_image_path)

      # save new image
      model_save_path = f'model_b_{train_batch_size}_{num_epochs}_{learn_rate}_{patch_dim}_{gap}_{epoch:04d}.pt'
      print('saving checkpoint', model_save_path)
      torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'global_trnloss': global_trn_loss,
            'global_valloss': global_val_loss
        }, model_save_path)



print("done")