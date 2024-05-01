# -*- coding: utf-8 -*-
print("hello")
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import librosa
import wandb

#wandb stuff
run = wandb.init(
    # set the wandb project where this run will be logged
    project="AudioSeparator",
    name = "1 GPU A100 10CHA 32CNN CONCAT HIGAMMA.9",
    #resume = True,

    # track hyperparameters and run metadata
    config={
    "architecture": "CNN",
    "dataset": "Synthetic Mixed Audio Data",
    "epochs": 50,
    }
)
print('wandb_initialized')

print("reading files")
files = []
for file in glob.glob("/scratch/09816/wertyuui345/ls6/AudioSeperation/training_data_additional/*.npz"):
    files.append(file)
for file in glob.glob("/scratch/09816/wertyuui345/ls6/AudioSeperation/training_data_final/*.npz"):
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
        data = beamfile['training_data']
        data = np.concatenate((data[:, 0, 0, :, :], data[:, 0, 1, :, :],data[:, 1, 1, :, :], data[:, 0, 2, :, :], data[:, 7, 1, :, :]), axis = 0)
        data = np.concatenate((data[0, :, :], data[1, :, :], data[2, :, :], data[3, :, :], data[4, :, :]), axis = 1)
        data = np.expand_dims(data, axis = 0)        
        data = np.asarray(data, dtype=complex)
        data = np.concatenate((data.real, data.imag), axis = 0)
        x = torch.from_numpy(np.asarray(data, dtype = float)).float()
        #print(x.shape)
        
        y = torch.from_numpy(beamfile['target'][:, 25])

        #print(x.shape)

        data = (x, y)

        if self.transform:
            data = self.transform(data)

        return data


train_dataset = BeamDataset(files, 24)
generator1 = torch.Generator().manual_seed(42)
train_set, valid_set = random_split(train_dataset, [0.9, 0.1], generator = generator1)
train_loader = DataLoader(train_set, batch_size=512,
                        shuffle=True, num_workers=16)
valid_loader = DataLoader(valid_set, batch_size=512, 
                          shuffle=False, num_workers=16)

print(len(train_dataset))

'''
Initial convolutional layers
Time Length = 50
Spectrogram Filter Banks = 257(?)
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
        #print(x.shape)
        #x = self.convLayer(x)
        x = self.asymStride(x)
        #print(x.shape)
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
        self.conv1=ConvLayer(2, 32) #8 input channels(change to # of beamspace input channels) Should be 15? I believe
        self.conv2=ConvLayer(32, 64)
        self.conv3=ConvLayer(64, 128)

        self.pool=nn.AvgPool2d((1,238)) #average along time axis (38)
        #Looks like the paper averages everything once and keeps freq the same
        self.flatten = nn.Flatten(start_dim=1)
        self.final1=FullCon(4224, 257) #Change 2048, to appropriate out size
        self.final2=Head(257, 257, nn.Sigmoid)

    def forward(self, x):
      x = self.conv1(x)
      #print(x.shape)
      x = self.conv2(x)
      #print(x.shape)
      x = self.conv3(x)
      #print(x.shape)
      x = self.pool(x)
      #print(x.shape)
      x = self.flatten(x)
      #print(x.shape)
      x = self.final1(x)
      #print(x.shape)
      x = self.final2(x)
      #print(x.shape)

      return x

model = BeamSep()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

def checkpoint(model, optimizer, filename, epoch, loss):
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, filename)

#if wandb.run.resumed:
#checkpoint = torch.load(wandb.restore('AudioMaskerCircular-MAG257.pth'))
#checkload = torch.load('AudioMaskerCircular-MAG257.pth')
#model.load_state_dict(checkload["model_state_dict"])
#optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    #epoch = checkpoint["epoch"]
    #loss = checkpoint["loss"]

device = None
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    model.to(device)
else:
    device = torch.device('cpu')
    
print(device)

#input = torch.randn((64, 24, 64, 15), dtype=torch.float).to(device)
#out = model(input)


#Training
for epoch in range(50):
    model.train()
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

    wandb.log({"Epoch": epoch + 1, "loss": running_loss / len(train_loader)})
    lr_scheduler.step()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')
    checkpoint(model, optimizer, f"checkpointmaglarge257_concat.pth", epoch, running_loss / len(train_loader))
        
    #Validate
    model.eval()
    total = 0
    running_vloss = 0.0
    with torch.no_grad():
        for i, data in enumerate(valid_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
        
            predicted = model(inputs)
            loss = criterion(predicted, labels)
            total += labels.size(0)

            running_vloss += loss.item()
            del inputs
            del labels

    lr_scheduler.step()
    wandb.log({"vloss": running_vloss / len(valid_loader)})
    print(f'Validation Loss: {running_vloss / len(valid_loader)}')

model_path = 'AudioMaskerCircular-MAG257_CONCAT.pth'
torch.save({'model_state_dict': model.state_dict(),
           'optimizer_state_dict': optimizer.state_dict()}, model_path)

print('Finished Training')

#net.load_state_dict(torch.load(model_path))
