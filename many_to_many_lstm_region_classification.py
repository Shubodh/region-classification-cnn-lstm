import argparse
import os
import shutil
import time

import torch 
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
#from logger import Logger

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

import pandas as pd
import pdb
import numpy as np
from PIL import Image


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
sequence_length = 28 #28
input_size = 28
hidden_size = 128
num_layers = 1
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01



model_resnet18 = torchvision.models.resnet18(num_classes=4)
model_resnet18 = torch.nn.DataParallel(model_resnet18).cuda()

checkpoint = torch.load("/home/shubodh/places365_training/trained_models/trained_models_places10_phase1/resnet18_best_phase1_4classes_unfrozen.pth.tar")
#    checkpoint = torch.load("/home/shubodh/places365_training/places365/trained_models_rapyuta4_phase2/resnet18_best_phase2_unfrozen_may25.pth.tar")
start_epoch = checkpoint['epoch']
best_prec = checkpoint['best_prec1']

model_resnet18.load_state_dict(checkpoint['state_dict'])
model_resnet18.module.fc = Identity()
model_resnet18.cuda()
print model_resnet18
cudnn.benchmark = True
abc = torch.randn(1,3,224,224)
output = model_resnet18(abc)
print "model_resnet18 after removing last layer {}".format(model_resnet18)
print "output {}".format(output)
print "output shape {}".format(output.shape)


transform = transforms.Compose([transforms.Resize([224,224]), transforms.RandomHorizontalFlip(),transforms.ToTensor()])

# Rapyuta Dataset
train_dataset = GetDataset(csv_file='/scratch/shubodh/places365/rapyuta4_classes/csv_data_labels/train_all.csv', root_dir='/scratch/shubodh/places365/rapyuta4_classes/train_all/', transform=transform)

test_dataset = GetDataset(csv_file='/scratch/shubodh/places365/rapyuta4_classes/csv_data_labels/test.csv', root_dir='/scratch/shubodh/places365/rapyuta4_classes/test_6900_7900/', transform=transform)

# Data loader

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler = ImbalancedDatasetSampler(train_dataset, csv_file='/scratch/shubodh/places365/rapyuta4_classes/csv_data_labels/train_all.csv', root_dir='/scratch/shubodh/places365/rapyuta4_classes/train_all/'), shuffle=True, num_workers=4) #Shuffle?

val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


# Many to Many LSTM  
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size 
        self.num_layers = num_layers 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) 
        self.fc = nn.Linear(hidden_size, num_classes) 
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM. ht = f(W1 * (h(t-1),x(t)))
        out, _ = self.lstm(x, (h0, c0))  

        # Last fc layer of all time steps. yt = W2 * ht
        out = self.fc(out[:, :, :]) 
        return out

model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
print model.parameters()
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #print ('images before reshaping {}'.format(images.shape)) #[100, 1, 28, 28]
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        #print ('images after reshaping {}'.format(images.shape)) #[100, 28, 28]
        #print ('labels size {}'.format(labels.shape))
        
        # Forward pass
        outputs = model(images)
        outputs = outputs.permute(0,2,1)
        #print outputs.shape
        #print labels.shape
        
        labels = labels.repeat(28,1)
        labels = labels.permute(1,0)
        #print labels.shape
        # k-dimensional loss (k=1 in this case of many to many, k=0 in many to one) 
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Optimizer
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in val_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
         
        #labels = labels.repeat(10,1)
        #labels = labels.permute(1,0)
         
        #print labels.shape
        outputs = model(images)
        #print outputs[:,-1].shape
        outputs = outputs.permute(0,2,1)
        #print outputs.shape
        #print labels.shape
        
        labels = labels.repeat(28,1)
        labels = labels.permute(1,0)
        
        _, predicted = torch.max(outputs.data, 1)
        total += (labels.size(0) * labels.size(1))
        correct += (predicted == labels).sum().item()
    print "correct {}".format(correct)
    print "total {}".format(total)
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) 

# Save the model checkpoint
torch.save(model.state_dict(), 'model_many_to_many_lstm.ckpt')
