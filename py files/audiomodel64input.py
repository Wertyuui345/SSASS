# -*- coding: utf-8 -*-
print("hello")
import glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import librosa

print("reading files")
files = []
for file in glob.glob("data/training_data/*.npz"):
    files.append(file)

print(len(files))

class BeamDataset(Dataset):
    def __init__(self, npz_path, channels, transform=None):
        self.transform = transform
        self.target_paths = npz_path #A list of all the .npz files in the dataset
        self.channels = channels

    def __len__(self):
        return len(self.target_paths)

    def __getitem__(self, idx):
        beamfile = np.load(self.target_paths[idx], allow_pickle=True) #load the ith .npz file
        # print(beamfile['training_data'][0])
        # print(beamfile['training_data'][1])
        data = np.moveaxis(abs(beamfile['training_data'][2].reshape(50, 257, 15))**2, (0, 1, 2), (2, 1, 0))
        #print(data)
            #beamspace transform, removing 0 HZ
        #x = torch.from_numpy(np.concatenate((x.real, x.imag), axis = 0)).float()
        x = torch.empty(self.channels, 64, 50)
        for channel in range(self.channels):
            #print(x[:,:, channel])
            x[channel,:,:] = torch.from_numpy(librosa.feature.melspectrogram(S = data[channel, :, :], sr = 44100, n_mels=64)).float() # PSD
       

        y = torch.from_numpy(beamfile['target'][25]).float()

        #print(x.shape)

        data = (x, y)

        if self.transform:
            data = self.transform(data)

        return data


train_dataset = BeamDataset(files, 15)
generator1 = torch.Generator().manual_seed(42)
train_set, valid_set = random_split(train_dataset, [0.8, 0.2], generator = generator1)
train_loader = DataLoader(train_set, batch_size=64,
                        shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_set, batch_size=64, 
                          shuffle=True, num_workers=0)

print(len(train_dataset))

"""#Model"""

'''
Initial convolutional layers
Time Length = 50
Spectrogram Filter Banks = 64
input Structure (Time, Frequency, Channels, Batch)
Needs to be Batch, Channel, Frequency, Time form to work
'''

class ConvLayer(nn.Module):
    def __init__(self, input_dim, output_dim, activation=nn.ReLU):
        super(ConvLayer, self).__init__()
        self.convLayer = nn.Sequential(
            nn.Conv2d(input_dim, input_dim, (3, 3), stride=(1,1), padding = (1,0), dtype=torch.float), #Frequency dim padding only not clear how much
            nn.BatchNorm2d(input_dim, track_running_stats=True),
            activation()
        )
        self.asymStride = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, (3, 3), stride=(2, 1), padding = (1,0), dtype=torch.float), #Frequency dim padding only not clear how much
            nn.BatchNorm2d(output_dim, track_running_stats=True),
            activation()
        )
    def forward(self, x):
        x = self.convLayer(x)
        x = self.asymStride(x)
        return x

class FullCon(nn.Module):
    def __init__(self, input_dim, output_dim, activation=nn.ReLU):
        super(FullCon, self).__init__()
        self.full = nn.Sequential(
            nn.Linear(input_dim, output_dim, dtype=torch.float),
            nn.BatchNorm1d(output_dim, track_running_stats=True),
            activation()
        )
    def forward(self, x):
        x = self.full(x)
        return x

class Head(nn.Module):
    def __init__(self, input_dim, output_dim, activation=nn.ReLU):
        super(Head, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, output_dim, dtype=torch.float),
            activation()
        )
    def forward(self, x):
        x = self.head(x)
        return x

class BeamSep(nn.Module):
    def __init__(self):
        super(BeamSep, self).__init__()
        self.conv1=ConvLayer(15, 16) #8 input channels(change to # of beamspace input channels) Should be 15? I believe
        self.conv2=ConvLayer(16, 32)
        self.conv3=ConvLayer(32, 64)

        self.pool=nn.AvgPool2d((1,38)) #average along time axis (38)
        #Looks like the paper averages everything once and keeps freq the same
        self.flatten = nn.Flatten(start_dim=1)
        self.final1=FullCon(512, 64) #Change 2048, to appropriate out size
        self.final2=Head(64, 257, nn.Sigmoid)

    def forward(self, x):
      x = self.conv1(x)
      x = self.conv2(x)
      x = self.conv3(x)

      x = self.pool(x)
      x = self.flatten(x)
      x = self.final1(x)
      x = self.final2(x)


      return x

model = BeamSep()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

def checkpoint(model, optimizer, filename, epoch, loss):
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, filename)

device = None
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    model.to(device)
else:
    device = torch.device('cpu')

input = torch.randn((64, 15, 64, 50), dtype=torch.float).to(device)
out = model(input)

prev = torch.load("AudioMasker64Input-Small.pth")
model.load_state_dict(prev['model_state_dict'])
optimizer.load_state_dict(prev['optimizer_state_dict'])

#Training
model.train()
for epoch in range(50):
    correct = 0
    total = 0
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        predicted = model(inputs)
        loss = criterion(predicted, labels)
        loss.backward()
        optimizer.step()
        total += labels.size(0)

        running_loss += loss.item()
        del inputs
        del labels

    lr_scheduler.step()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')
    if (epoch % 20 == 0 and epoch != 0):
        checkpoint(model, optimizer, f"epoch-{epoch}.pth", epoch, running_loss / len(train_loader))

print('Finished Training')

model.eval()

#Validate
total = 0
running_loss = 0.0
for i, data in enumerate(valid_loader, 0):
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    
    predicted = model(inputs)
    loss = criterion(predicted, labels)
    total += labels.size(0)

    running_loss += loss.item()
    del inputs
    del labels

#lr_scheduler.step()
print(f'Validation Loss: {running_loss / len(valid_loader)}')

model_path = 'AudioMasker64Input-Small.pth'
torch.save({'model_state_dict': model.state_dict(),
           'optimizer_state_dict': optimizer.state_dict()}, model_path)


#net.load_state_dict(torch.load(model_path))
