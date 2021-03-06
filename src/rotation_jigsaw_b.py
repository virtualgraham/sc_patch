import random
import time
from glob import glob
import math
import os

from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from torchsummary import summary
from tqdm import tqdm # progress bar


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Rotation
# Saliency Check
# With color shift or grayscale conversion

#########################################
# Parameters 
#########################################

training_image_paths = glob('/data/mini-open-images-dataset/train/*.jpg')
validation_image_paths = glob('/data/mini-open-images-dataset/validation/*.jpg')

train_dataset_length = 40960
validation_dataset_length = 40960
train_batch_size = 128
validation_batch_size = 128
num_epochs = 3000
backup_after_epochs = 10 
model_save_prefix = "rotation_jigsaw_b"
color_shift = 1
patch_dim = 128
jitter = 22
gray_portion = .30
reuse_image_count = 4

learn_rate = 0.001
momentum = 0.974
weight_decay = 0.0005



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

class ShufflePatchDataset(Dataset):

  def __init__(self, image_paths, patch_dim, length, jitter, color_shift, transform=None):
    self.image_paths = image_paths
    self.patch_dim = patch_dim
    self.length = length
    self.jitter = jitter
    self.color_shift = color_shift
    self.transform = transform
    self.image_reused = 0

    self.sub_window_width = self.patch_dim + 2*self.jitter + 2*self.color_shift
    self.window_width = 2*self.sub_window_width
    
    # self.min_image_width = self.window_width

  def __len__(self):
    return self.length

  def random_jitter(self):
    return int(math.floor((self.jitter * 2 * random.random())))

  def random_shift(self):
    return random.randrange(self.color_shift * 2 + 1)

  def prep_patch(self, image):
 
    cropped = np.empty((self.patch_dim, self.patch_dim, 3), dtype=np.uint8)

    if(random.random() < gray_portion):

      pil_patch = Image.fromarray(image)
      pil_patch = pil_patch.convert('L')
      pil_patch = pil_patch.convert('RGB')
      np.copyto(cropped, np.array(pil_patch)[self.color_shift:self.color_shift+self.patch_dim, self.color_shift:self.color_shift+self.patch_dim, :])
      
    else:

      shift = [self.random_shift() for _ in range(6)]
      cropped[:,:,0] = image[shift[0]:shift[0]+self.patch_dim, shift[1]:shift[1]+self.patch_dim, 0]
      cropped[:,:,1] = image[shift[2]:shift[2]+self.patch_dim, shift[3]:shift[3]+self.patch_dim, 1]
      cropped[:,:,2] = image[shift[4]:shift[4]+self.patch_dim, shift[5]:shift[5]+self.patch_dim, 2]

    return cropped


  def __getitem__(self, index):
    # [y, x, chan], dtype=uint8, top_left is (0,0)
    
    image_index = int(math.floor((len(self.image_paths) * random.random())))
    
    if self.image_reused == 0:
      pil_image = Image.open(self.image_paths[image_index]).convert('RGB')
      # if pil_image.size[1] > pil_image.size[0]:
      #   self.pil_image = pil_image.resize((self.min_image_width, int(round(pil_image.size[1]/pil_image.size[0] * self.min_image_width))))
      # else:
      #   self.pil_image = pil_image.resize((int(round(pil_image.size[0]/pil_image.size[1] * self.min_image_width)), self.min_image_width))
      self.image_reused = reuse_image_count - 1

    else:
      self.image_reused -= 1

    image = np.array(self.pil_image)
    
    window_y_coord = int(math.floor((image.shape[0] - self.window_width) * random.random()))
    window_x_coord = int(math.floor((image.shape[1] - self.window_width) * random.random()))

    window = image[window_y_coord:window_y_coord+self.window_width, window_x_coord:window_x_coord+self.window_width]
    
    rotation_label = int(math.floor((4 * random.random())))
    order_label = int(math.floor((24 * random.random()))) 
    
    if rotation_label>0:
      window = np.rot90(window, rotation_label).copy()

    patch_coords = [
      (self.random_jitter(), self.random_jitter()),
      (self.random_jitter(), self.sub_window_width + self.random_jitter()),
      (self.sub_window_width + self.random_jitter(), self.random_jitter()),
      (self.sub_window_width + self.random_jitter(), self.sub_window_width + self.random_jitter()),
    ]

    patch_coords = [pc for _,pc in sorted(zip(patch_order_arr[order_label],patch_coords))]

    patch_a = window[patch_coords[0][0]:patch_coords[0][0]+self.patch_dim+2*self.color_shift, patch_coords[0][1]:patch_coords[0][1]+self.patch_dim+2*self.color_shift]
    patch_b = window[patch_coords[1][0]:patch_coords[1][0]+self.patch_dim+2*self.color_shift, patch_coords[1][1]:patch_coords[1][1]+self.patch_dim+2*self.color_shift]
    patch_c = window[patch_coords[2][0]:patch_coords[2][0]+self.patch_dim+2*self.color_shift, patch_coords[2][1]:patch_coords[2][1]+self.patch_dim+2*self.color_shift]
    patch_d = window[patch_coords[3][0]:patch_coords[3][0]+self.patch_dim+2*self.color_shift, patch_coords[3][1]:patch_coords[3][1]+self.patch_dim+2*self.color_shift]

    patch_a = self.prep_patch(patch_a)
    patch_b = self.prep_patch(patch_b)
    patch_c = self.prep_patch(patch_c)
    patch_d = self.prep_patch(patch_d)

    combined_label = np.array(rotation_label * 24 + order_label).astype(np.int64)
        
    if self.transform:
      patch_a = self.transform(patch_a)
      patch_b = self.transform(patch_b)
      patch_c = self.transform(patch_c)
      patch_d = self.transform(patch_d)

    return patch_a, patch_b, patch_c, patch_d, combined_label
    

##################################################
# Creating Train/Validation dataset and dataloader
##################################################

traindataset = ShufflePatchDataset(training_image_paths, patch_dim, train_dataset_length, jitter, color_shift,
                         transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))

trainloader = torch.utils.data.DataLoader(traindataset, 
                                          batch_size=train_batch_size,
                                          num_workers=4,
                                          shuffle=False)


valdataset = ShufflePatchDataset(validation_image_paths, patch_dim, validation_dataset_length, jitter, color_shift,
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
        nn.Linear(512 * 3 * 3, 4096),
        nn.ReLU(True),
        nn.Dropout(),
      )

      self.fc = nn.Sequential(
        nn.Linear(4*4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 96),
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

    return output

model = VggNetwork().to(device)
summary(model, [(3, patch_dim, patch_dim), (3, patch_dim, patch_dim), (3, patch_dim, patch_dim), (3, patch_dim, patch_dim)])



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

training_image_paths = glob(f'{model_save_prefix}_*.pt')

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

print("starting train loop (-.-)")

for epoch in range(last_epoch+1, num_epochs):
    print("epoch", epoch)

    train_running_loss = []
    
    start_time = time.time()

    model.train()


    ## Train 
    for idx, data in tqdm(enumerate(trainloader), total=int(len(traindataset)/train_batch_size)):
        patch_a, patch_b, patch_c, patch_d, patch_shuffle_order_label = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)
        optimizer.zero_grad()
        output = model(patch_a, patch_b, patch_c, patch_d)
        loss = criterion(output, patch_shuffle_order_label)
        loss.backward()
        optimizer.step()
        
        train_running_loss.append(loss.item())
  
    global_trn_loss.append(sum(train_running_loss) / len(train_running_loss))


    ## Validation

    if epoch % backup_after_epochs == 0:
      val_running_loss = []
      correct = 0
      total = 0
      model.eval()

      with torch.no_grad():
        for idx, data in tqdm(enumerate(valloader), total=int(len(valdataset)/validation_batch_size)):
          patch_a, patch_b, patch_c, patch_d, patch_shuffle_order_label = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)
          output = model(patch_a, patch_b, patch_c, patch_d)
          loss = criterion(output, patch_shuffle_order_label)
          val_running_loss.append(loss.item())
        
          _, predicted = torch.max(output.data, 1)
          total += patch_shuffle_order_label.size(0)
          correct += (predicted == patch_shuffle_order_label).sum()
        print('Val Progress --- total:{}, correct:{}'.format(total, correct.item()))
        print('Val Accuracy of the network on the test images: {}%'.format(100 * correct.item() / total))

      global_val_loss.append(sum(val_running_loss) / len(val_running_loss))

    else:
      if len(global_val_loss) > 0:
        global_val_loss.append(global_val_loss[-1])
      else:
        global_val_loss.append(0)
    
    
    print('Epoch [{}/{}], TRNLoss:{:.4f}, VALLoss:{:.4f}, Time:{:.2f}'.format(
        epoch + 1, num_epochs, global_trn_loss[-1], global_val_loss[-1],
        (time.time() - start_time) / 60))
    
    # delete old images
    training_image_paths = glob(f'{model_save_prefix}_*.pt')
    if len(training_image_paths) > 2:
      training_image_paths.sort()
      for i in range(len(training_image_paths)-2):
        training_image_path = training_image_paths[i]
        os.remove(training_image_path)

    # save new image
    model_save_path = f'{model_save_prefix}_{epoch:04d}.pt'
    print('saving checkpoint', model_save_path)
    torch.save(
      {
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'loss': loss,
          'global_trnloss': global_trn_loss,
          'global_valloss': global_val_loss
      }, model_save_path
    )
  
    if epoch % backup_after_epochs == 0:
      print('backing up checkpoint', model_save_path)
      os.system(f'aws s3 cp /data/{model_save_path} s3://guiuan/{model_save_prefix}_{epoch:04d}_{learn_rate}_{global_trn_loss[-1]:.4f}_{(100 * correct.item()/total):.2f}.pt')

print("done")