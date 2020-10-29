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
train_dataset_length = 40192
validation_dataset_length = 2048
gap = 48
jitter = 7
train_batch_size = 256
validation_batch_size = 128
num_epochs = 1500

learn_rate = 0.001
momentum = 0.999
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

patch_order_arr = [
  (0, 1, 2, 3),
  (0, 1, 3, 2),
  (0, 2, 1, 3),
  (0, 2, 3, 1),
  (0, 3, 1, 2),
  (0, 3, 2, 1),
  (1, 0, 2, 3),
  (1, 0, 3, 2),
  (1, 2, 0, 3),
  (1, 2, 3, 0),
  (1, 3, 0, 2),
  (1, 3, 2, 0),
  (2, 0, 1, 3),
  (2, 0, 3, 1),
  (2, 1, 0, 3),
  (2, 1, 3, 0),
  (2, 3, 0, 1),
  (2, 3, 1, 0),
  (3, 0, 1, 2),
  (3, 0, 2, 1),
  (3, 1, 0, 2),
  (3, 1, 2, 0),
  (3, 2, 0, 1),
  (3, 2, 1, 0)
]

class MyDataset(Dataset):

  def __init__(self, image_paths, patch_dim, length, gap, jitter, transform=None):
    self.image_paths = image_paths
    self.patch_dim = patch_dim
    self.length = length
    self.gap = gap
    self.jitter = jitter
    self.transform = transform
    self.margin = math.ceil((2*patch_dim + 2*jitter + gap)/2)
    self.min_width = 2 * self.margin + 1

  def __len__(self):
    return self.length
  
  def half_gap(self):
    return math.ceil(self.gap/2)

  def random_jitter(self):
    return int(math.floor((self.jitter * 2 * random.random()))) - self.jitter

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
    if (image.shape[0] - self.min_width) <= 0 or (image.shape[1] - self.min_width) <= 0:
        return self.__getitem__(index)
    
    center_y_coord = int(math.floor((image.shape[0] - self.margin*2) * random.random())) + self.margin
    center_x_coord = int(math.floor((image.shape[1] - self.margin*2) * random.random())) + self.margin

    patch_coords = [
      (
        center_y_coord - (self.patch_dim + self.half_gap() + self.random_jitter()),
        center_x_coord - (self.patch_dim + self.half_gap() + self.random_jitter())
      ),
      (
        center_y_coord - (self.patch_dim + self.half_gap() + self.random_jitter()),
        center_x_coord + self.half_gap()+ self.random_jitter()
      ),
      (
        center_y_coord + self.half_gap() + self.random_jitter(),
        center_x_coord - (self.patch_dim + self.half_gap() + self.random_jitter())
      ),
      (
        center_y_coord + self.half_gap() + self.random_jitter(),
        center_x_coord + self.half_gap() + self.random_jitter()
      )
    ]
    
    patch_shuffle_order_label = int(math.floor((24 * random.random())))

    patch_coords = [pc for _,pc in sorted(zip(patch_order_arr[patch_shuffle_order_label],patch_coords))]

    patch_a = image[patch_coords[0][0]:patch_coords[0][0]+self.patch_dim, patch_coords[0][1]:patch_coords[0][1]+self.patch_dim]
    patch_b = image[patch_coords[1][0]:patch_coords[1][0]+self.patch_dim, patch_coords[1][1]:patch_coords[1][1]+self.patch_dim]
    patch_c = image[patch_coords[2][0]:patch_coords[2][0]+self.patch_dim, patch_coords[2][1]:patch_coords[2][1]+self.patch_dim]
    patch_d = image[patch_coords[3][0]:patch_coords[3][0]+self.patch_dim, patch_coords[3][1]:patch_coords[3][1]+self.patch_dim]

    self.prep_patch(patch_a)
    self.prep_patch(patch_b)
    self.prep_patch(patch_c)
    self.prep_patch(patch_d)

    patch_shuffle_order_label = np.array(patch_shuffle_order_label).astype(np.int64)
        
    if self.transform:
      patch_a = self.transform(patch_a)
      patch_b = self.transform(patch_b)
      patch_c = self.transform(patch_c)
      patch_d = self.transform(patch_d)

    return patch_a, patch_b, patch_c, patch_d, patch_shuffle_order_label



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
    
      self.fc6 = nn.Sequential(
        nn.Linear(512 * 3 * 3, 4096),
        nn.ReLU(True),
        nn.Dropout(),
      )

      self.fc = nn.Sequential(
        nn.Linear(4*4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 24),
      )

  def forward_once(self, x):
    output= self.cnn(x)
    output = output.view(output.size()[0], -1)
    output = self.fc6(output)
    return output

  def forward(self, patch_a, patch_b, patch_c, patch_d):
    output_fc6_patch_a = self.forward_once(patch_a)
    output_fc6_patch_b = self.forward_once(patch_b)
    output_fc6_patch_c = self.forward_once(patch_c)
    output_fc6_patch_d = self.forward_once(patch_d)

    output = torch.cat((output_fc6_patch_a, output_fc6_patch_b, output_fc6_patch_c, output_fc6_patch_d), 1)
    output = self.fc(output)

    return output, output_fc6_patch_a, output_fc6_patch_b, output_fc6_patch_c, output_fc6_patch_d

model = AlexNetwork().to(device)
summary(model, [(3, 96, 96), (3, 96, 96), (3, 96, 96), (3, 96, 96)])



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



#############################################
# Load Checkpoint
#############################################

global_trn_loss = []
global_val_loss = []

last_epoch = -1

training_image_paths = glob(f'model_{train_batch_size}_{num_epochs}_{learn_rate}_{patch_dim}_{gap}_*.pt')

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
        patch_a, patch_b, patch_c, patch_d, patch_shuffle_order_label = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)
        # print(uniform_patch.size(), random_patch.size())
        optimizer.zero_grad()
        output, output_fc6_patch_a, output_fc6_patch_b, output_fc6_patch_c, output_fc6_patch_d = model(patch_a, patch_b, patch_c, patch_d)
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
          patch_a, patch_b, patch_c, patch_d, patch_shuffle_order_label = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)
          output, output_fc6_patch_a, output_fc6_patch_b, output_fc6_patch_c, output_fc6_patch_d = model(uniform_patch, random_patch)
          loss = criterion(output, patch_shuffle_order_label)
          val_running_loss.append(loss.item())
        
          _, predicted = torch.max(output.data, 1)
          total += patch_shuffle_order_label.size(0)
          correct += (predicted == patch_shuffle_order_label).sum()
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
      training_image_paths = glob(f'model_{train_batch_size}_{num_epochs}_{learn_rate}_{patch_dim}_{gap}_*.pt')
      if len(training_image_paths) > 2:
        training_image_paths.sort()
        for i in range(len(training_image_paths)-2):
          training_image_path = training_image_paths[i]
          os.remove(training_image_path)

      # save new image
      model_save_path = f'model_{train_batch_size}_{num_epochs}_{learn_rate}_{patch_dim}_{gap}_{epoch:04d}.pt'
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