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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group('nccl')
rank = dist.get_rank()
print('Rank: ', rank)
if rank == 0:
#wandb stuff
    wandb.init(
        # set the wandb project where this run will be logged
        project="AudioSeparator",
        name = "1 GPU A100 Actual MAG",

        # track hyperparameters and run metadata
        config={
        "architecture": "CNN",
        "dataset": "Synthetic Mixed Audio Data",
        "epochs": 200,
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
        data = np.concatenate((data[:, :, 0, :, :], data[:, :, 1, :, :],data[:, :, 2, :, :]), axis = 1)
        x = abs(np.squeeze(data))**2

        x = torch.from_numpy(np.asarray(data, dtype=float)).float()


        y = torch.from_numpy(beamfile['target'][:, 25])

        #print(x.shape)

        data = (x, y)

        if self.transform:
            data = self.transform(data)

        return data


train_dataset = BeamDataset(files, 24)
generator1 = torch.Generator().manual_seed(42)
train_set, valid_set = random_split(train_dataset, [0.9, 0.1], generator = generator1)
                        
num_tasks = dist.get_world_size()
trainsampler = torch.utils.data.DistributedSampler(train_set, num_replicas=num_tasks, rank=rank, shuffle = True) 
validsampler = torch.utils.data.DistributedSampler(valid_set, num_replicas=num_tasks, rank=rank, shuffle = False)

train_loader = DataLoader(train_set, batch_size=512, sampler = trainsampler, num_workers=16)
valid_loader = DataLoader(valid_set, batch_size=512, sampler = validsampler, num_workers=16)

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
        self.conv1=ConvLayer(48, 16) #8 input channels(change to # of beamspace input channels) Should be 15? I believe
        self.conv2=ConvLayer(16, 32)
        self.conv3=ConvLayer(32, 64)

        #self.pool=nn.AvgPool2d((1,1)) #average along time axis (38)
        #Looks like the paper averages everything once and keeps freq the same
        self.flatten = nn.Flatten(start_dim=1)
        self.final1=FullCon(2112, 257) #Change 2048, to appropriate out size
        self.final2=Head(257, 257, nn.Sigmoid)

    def forward(self, x):
      x = self.conv1(x)
      #print(x.shape)
      x = self.conv2(x)
      #print(x.shape)
      x = self.conv3(x)
      #print(x.shape)
      #x = self.pool(x)
      #print(x.shape)
      x = self.flatten(x)
      #print(x.shape)
      x = self.final1(x)
      #print(x.shape)
      x = self.final2(x)
      #print(x.shape)

      return x


def checkpoint(model, optimizer, filename, epoch, loss):
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, filename)

model = BeamSep()
device = None
if torch.cuda.is_available():
    device_id = rank % torch.cuda.device_count()
    print(torch.cuda.device_count())
    model.to(device_id)
    model = DDP(model, device_ids=[device_id])
else:
    device = torch.device('cpu')
    model.to(device)

print(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

#input = torch.randn((64, 24, 64, 15), dtype=torch.float).to(device)
#out = model(input)

# prev = torch.load("AudioMasker-Small.pth")
# model.load_state_dict(prev['model_state_dict'])
# optimizer.load_state_dict(prev['optimizer_state_dict'])

#Training
for epoch in range(200):
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
    lr_scheduler.step()
    if rank == 0:
        wandb.log({"Epoch": epoch + 1, "loss": running_loss / len(train_loader)})
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')
        checkpoint(ddp_model, optimizer, f"checkpoint1ActualMAG.pth", epoch, running_loss / len(train_loader))
        
    #Validate
    model.eval()
    total = 0
    running_vloss = 0.0
    for i, data in enumerate(valid_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        predicted = model(inputs)
        loss = criterion(predicted, labels)
        total += labels.size(0)

        running_vloss += loss.item()
        del inputs
        del labels

    #lr_scheduler.step()
    if rank == 0:
        wandb.log({"vloss": running_vloss / len(valid_loader)})
        print(f'Validation Loss: {running_vloss / len(valid_loader)}')
if rank == 0:
    model_path = 'AudioMaskerCircular1Actual-SmallMAG.pth'
    torch.save({'model_state_dict': model.state_dict(),
           'optimizer_state_dict': optimizer.state_dict()}, model_path)

dist.destroy_process_group()
print('Finished Training')


#net.load_state_dict(torch.load(model_path))
