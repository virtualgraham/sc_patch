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

training_image_paths = glob('/data/open-images-dataset/train/*.jpg')
validation_image_paths = glob('/data/open-images-dataset/validation/*.jpg')

permutations = np.load("src/nine_patch/permutations_1000.npy")

train_dataset_length = 40192 # 314 iterations
validation_dataset_length = 2048 
train_batch_size = 256
validation_batch_size = 256
num_epochs = 1500
save_after_epochs = 1 
backup_after_epochs = 10 
model_save_prefix = "nine_patch_a"
permutation_count = 1000

patch_dim = 64
jitter = 11 # gap = t* jitter
gray_portion = .30

learn_rate = 0.001
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

  def __init__(self, image_paths, patch_dim, length, jitter, transform=None):
    self.image_paths = image_paths
    self.patch_dim = patch_dim
    self.length = length
    self.gap = 2*jitter
    self.jitter = jitter
    self.color_shift = 2
    self.transform = transform

    self.sub_window_width = self.patch_dim + 2*self.jitter + 2*self.color_shift
    self.window_width = 3*self.sub_window_width
    
    self.min_image_width = self.window_width + 1

  def __len__(self):
    return self.length
  
  def half_gap(self):
    return self.jitter

  def random_jitter(self):
    return int(math.floor(self.jitter * 2 * random.random()))

  def random_shift(self):
    return random.randrange(self.color_shift * 2 + 1)

  # crops the patch by self.color_shift on each side
  def prep_patch(self, sub_window, gray):
 
    cropped = np.empty((self.patch_dim, self.patch_dim, 3), dtype=np.uint8)

    if(gray):

      pil_patch = Image.fromarray(sub_window)
      pil_patch = pil_patch.convert('L')
      pil_patch = pil_patch.convert('RGB')
      np.copyto(cropped, np.array(pil_patch)[self.color_shift:self.color_shift+self.patch_dim, self.color_shift:self.color_shift+self.patch_dim, :])
      
    else:

      shift = [self.random_shift() for _ in range(6)]
      cropped[:,:,0] = sub_window[shift[0]:shift[0]+self.patch_dim, shift[1]:shift[1]+self.patch_dim, 0]
      cropped[:,:,1] = sub_window[shift[2]:shift[2]+self.patch_dim, shift[3]:shift[3]+self.patch_dim, 1]
      cropped[:,:,2] = sub_window[shift[4]:shift[4]+self.patch_dim, shift[5]:shift[5]+self.patch_dim, 2]

    return cropped


  def __getitem__(self, index):
    # [y, x, chan], dtype=uint8, top_left is (0,0)
        
    image_index = int(math.floor((len(self.image_paths) * random.random())))
    pil_image = Image.open(self.image_paths[image_index]).convert('RGB')
    image = np.array(pil_image)

    # If image is too small, try another image
    if (image.shape[0] - self.min_image_width) <= 0 or (image.shape[1] - self.min_image_width) <= 0:
        return self.__getitem__(index)
    
    window_y_coord = int(math.floor((image.shape[0] - self.window_width) * random.random()))
    window_x_coord = int(math.floor((image.shape[1] - self.window_width) * random.random()))

    sub_window_coords = [
      (window_y_coord, window_x_coord),
      (window_y_coord, window_x_coord + self.sub_window_width),
      (window_y_coord, window_x_coord + 2 * self.sub_window_width),
      (window_y_coord + self.sub_window_width, window_x_coord),
      (window_y_coord + self.sub_window_width, window_x_coord + self.sub_window_width),
      (window_y_coord + self.sub_window_width, window_x_coord + 2 * self.sub_window_width),
      (window_y_coord + 2 * self.sub_window_width, window_x_coord),
      (window_y_coord + 2 * self.sub_window_width, window_x_coord + self.sub_window_width),
      (window_y_coord + 2 * self.sub_window_width, window_x_coord + 2 * self.sub_window_width)
    ]

    # top left corner of each patch before shifting color channels and cropping
    uncropped_patch_coords = [(y+self.random_jitter()+self.color_shift, x+self.random_jitter()+self.color_shift) for (y,x) in sub_window_coords]
    
    permutation_index = int(math.floor((permutation_count * random.random())))

    uncropped_patch_coords = [pc for _,pc in sorted(zip(permutations[permutation_index],uncropped_patch_coords))]

    uncropped_patches = [image[y:y+self.patch_dim+2*self.color_shift, x:x+self.patch_dim+2*self.color_shift] for (y,x) in uncropped_patch_coords]

    gray = random.random() < gray_portion
    patches = [self.prep_patch(patch, gray) for patch in uncropped_patches]

    if self.transform:
      patches = [self.transform(patch) for patch in patches]

    return patches, np.array(permutation_index).astype(np.int64)



##################################################
# Creating Train/Validation dataset and dataloader
##################################################

traindataset = ShufflePatchDataset(training_image_paths, patch_dim, train_dataset_length, jitter,
                         transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

trainloader = torch.utils.data.DataLoader(traindataset, 
                                          batch_size=train_batch_size,
                                          num_workers=4,
                                          shuffle=False)


valdataset = ShufflePatchDataset(validation_image_paths, patch_dim, validation_dataset_length, jitter,
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
        nn.Linear(512 * 2 * 2, 1024),
        nn.ReLU(True),
        nn.Dropout(),
      )

      self.fc = nn.Sequential(
        nn.Linear(9*1024, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, permutation_count),
      )

  def forward_once(self, x):
    output= self.cnn(x)
    output = output.view(output.size()[0], -1)
    output = self.fc6(output)
    return output

  def forward(self, *patches):
    output = torch.cat([self.forward_once(patch) for patch in patches], 1)
    return self.fc(output)

model = VggNetwork().to(device)
summary(model, [(3, patch_dim, patch_dim) for _ in range(9)])



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

training_image_paths = glob(f'{model_save_prefix}_{train_batch_size}_{num_epochs}_{learn_rate}_{patch_dim}_{2*jitter}_*.pt')

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

        patches = [d.to(device) for d in data[0:9]]
        permutation_index = data[9].to(device)

        optimizer.zero_grad()
        output = model(patches...)
        loss = criterion(output, permutation_index)
        loss.backward()
        optimizer.step()

        train_running_loss.append(loss.item())
    else:
      correct = 0
      total = 0
      model.eval()
      with torch.no_grad():
        for idx, data in tqdm(enumerate(valloader), total=int(len(valdataset)/validation_batch_size)):

          patches = [d.to(device) for d in data[0:9]]
          permutation_index = data[9].to(device)

          output = model(patches...)
          loss = criterion(output, permutation_index)
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
      training_image_paths = glob(f'{model_save_prefix}_{train_batch_size}_{num_epochs}_{learn_rate}_{patch_dim}_{2*jitter}_*.pt')
      if len(training_image_paths) > 2:
        training_image_paths.sort()
        for i in range(len(training_image_paths)-2):
          training_image_path = training_image_paths[i]
          os.remove(training_image_path)

      # save new image
      model_save_path = f'{model_save_prefix}_{train_batch_size}_{num_epochs}_{learn_rate}_{patch_dim}_{2*jitter}_{epoch:04d}.pt'
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
    
      if epoch % backup_after_epochs == 0:
        print('backing up checkpoint', model_save_path)
        os.system(f'aws s3 cp /data/{model_save_path} s3://guiuan/{model_save_prefix}_{epoch:04d}_{learn_rate}_{global_trn_loss[-1]:.4f}_{(100 * correct.item()/total):.2f}.pt')

print("done")