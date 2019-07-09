import argparse
import os
import shutil
import time
import sys

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
from sklearn import preprocessing

import pandas as pd
import pdb
import numpy as np
from PIL import Image

from scipy.stats import mode 

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


# Hyper-parameters
sequence_length = 20 #28
input_size = 512
hidden_size = 128
num_layers = 1
num_classes = 4 
batch_size = 100
num_epochs = 5
learning_rate = 0.01


transform = transforms.Compose(transforms.ToTensor())

resnet18 = 'resnet18'

#logger_train = Logger('./logs_unfrozen/train_lr_0.001_0.0001')
#logger_val = Logger('./logs_unfrozen/val_lr_0.001_0.0001')

def main():
    global best_prec1, lr_all, lr_fc
    

    train_dataset = GetDataset(csv_file='/scratch/shubodh/places365/rapyuta4_classes/csv_data_labels/train_all_21219.csv', root_dir='/home/shubodh/places365_training/region-classification-cnn-lstm/npy/', sequence_length=sequence_length, transform=transform)
    test_temp_dataset = GetTempTestDataset(csv_file='/scratch/shubodh/places365/rapyuta4_classes/csv_data_labels/test_8419.csv', root_dir='/home/shubodh/places365_training/region-classification-cnn-lstm/npy/', sequence_length=sequence_length, transform=transform)
    test_dataset = GetTestDataset(csv_file='/scratch/shubodh/places365/rapyuta4_classes/csv_data_labels/test_8419.csv', root_dir='/home/shubodh/places365_training/region-classification-cnn-lstm/npy/', sequence_length=sequence_length, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler= ImbalancedDatasetSampler(train_dataset, csv_file='/scratch/shubodh/places365/rapyuta4_classes/csv_data_labels/train_all_21219.csv', root_dir='/home/shubodh/places365_training/region-classification-cnn-lstm/npy/', sequence_length=sequence_length), shuffle=False, num_workers=4)
    #dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
    val_temp_loader = DataLoader(dataset=test_temp_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

#    ab = 0
#    for images, labels in val_loader:
#        #output_resnet = images.reshape(-1, sequence_length, input_size).to(device)
#        #print "output resnet shape {}".format(output_resnet.shape)
#        #labels = labels.to(device)
#        print "labels {}".format(labels)
#        print ab
#        ab += 1
    
    #w = torch.Tensor([0.46,0.56,5.42,4.69]).cuda()
    #criterion = torch.nn.CrossEntropyLoss(weight=w)
   
#
    model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        #adjust_learning_rate(optimizer, epoch)
        #learning rate decay every 30 epochs
        #lr_all = lr_all * (0.1 ** ((epoch - 115) // 30))
       
        for i, (images, labels) in enumerate(train_loader):
            # train for one epoch
            labels = labels.to(device)
            labels = labels.long()
            #print "input shape before {}".format(input_var.shape)
            
            # compute output
            output_resnet = images.reshape(-1, sequence_length, input_size).to(device)
            output_resnet = output_resnet.float()
            print "output_resnet shape train: {}".format(output_resnet.shape)
            # Forward pass LSTM
            outputs = model(output_resnet)
            outputs = outputs.permute(0,2,1)
            #print "outputs shape {}".format(outputs.shape)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Optimizer
            optimizer.step()
            
            if (i+1) % 1 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    #model_saved = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
    #model_saved.load_state_dict(torch.load("./model_many_to_many_lstm.ckpt"))
    #model_saved.load_state_dict(torch.load("./model_many_to_many_lstm_5epochs_satya.ckpt", map_location=lambda storage, loc: storage))
#    
    # Test the model
    with torch.no_grad():
        output_vector = np.zeros([8400,20,4])
        out_i = 0
        for images, labels in val_temp_loader:
            #images = np.array(images)
            #for i in range(images.shape[0]):
            #    images[i] = preprocessing.normalize(images[i,:,:], norm='l2')
            #images = torch.from_numpy(images)
            output_resnet = images.reshape(-1, sequence_length, input_size).to(device)
            output_resnet = output_resnet.float()
            outputs = model(output_resnet)
            
            outputs_np = outputs.data.cpu().numpy()
            output_vector[out_i*100:((out_i+1)*100),:,:] = outputs_np[:,:,:]
            #print " output vector {}shape {} ".format(output_vector, output_vector.shape)
            print out_i
            out_i += 1
            
        output_vector_indi = np.zeros([8419,20,4])
        for i in range(output_vector_indi.shape[0]):
            for k in range(output_vector_indi.shape[1]):
                if ((i - (19 - k)) >= 0 and (i - (19 - k)) < 8400):
                    output_vector_indi[i,k,:] = output_vector[i-(19-k),19-k,:]
        #DEBUG: the matrix should look like a parallelogram: rectange with top right corner pulled upwards
        #print "output_vector_indi starting{}".format(output_vector_indi[0:2,:,:])
        #print "output_vector_indi ending{}".format(output_vector_indi[8417:8419,:,:])
        
        output_vector_final = np.average(output_vector_indi, axis=1)
        print "output_vector_final {}".format(output_vector_final.shape)
       
        output_final = torch.from_numpy(output_vector_final)
        output_final = output_final.to(device)
        correct = 0
        total = 0
        pre_i = 0
        true_list = np.array([])
        pred_list = np.array([])
        for images, labels in val_loader:
            _, predicted = torch.max(output_final.data, 1)
            #print "predicted {}".format(predicted)
            #print "labels {}".format(labels)
            labels = labels.to(device)
            labels = labels.long()
            total += (labels.size(0))
            #predicted = predicted.long()
            correct += (predicted[pre_i*100:((pre_i+1)*100)] == labels).sum().item()

            labels_temp = np.asarray(labels.data.cpu().numpy())
            true_list = np.append(true_list, labels_temp)
            pred_list = np.append(pred_list, predicted[pre_i*100:((pre_i+1)*100)].cpu().numpy())

            pre_i += 1

        torch.save(model.state_dict(), 'model_many_to_many_lstm_5epochs_finale.ckpt')

        print "correct {}".format(correct)
        print "total {}".format(total)
        print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total)) 

        print('Confusion matrix: ')
        print confusion_matrix(true_list, pred_list)
        print('Accuracy score: ', accuracy_score(true_list, pred_list))
        print(classification_report(true_list,pred_list))


#
#    # Save the model checkpoint
   # torch.save(model.state_dict(), 'model_many_to_many_lstm_5epochs_finale.ckpt')

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        
        # measure accuracy and record loss
        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top3.update(prec3.item(), input.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top3=top3))

        
        
    # TENSORBOARD LOGGING
    # 1. Log scalar values (scalar summary)
    info = { 'loss': losses.avg, 'accuracy': top1.avg }

    for tag, value in info.items():
        logger_train.scalar_summary(tag, value, epoch)

    # 2. Log values and gradients of the parameters (histogram summary)
    for tag, value in model.named_parameters():
        tag = tag.replace('.', '/')
        logger_train.histo_summary(tag, value.data.cpu().numpy(), epoch)
        logger_train.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch)

    # 3. Log training images (image summary)
    info = { 'train_images': input_var.view(-1, 28, 28)[:10].cpu().numpy() }

    for tag, input_var in info.items():
        logger_train.image_summary(tag, input_var, epoch) 


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    true_list = np.array([])
    pred_list = np.array([])
    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)
            
            target_ew = torch.autograd.Variable(target.squeeze()).cuda()
            #print(target_ew.size())
            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)
            # adding custom validation metrics from sklearn
            target_ew_np = np.asarray(target_ew.data.cpu().numpy())
            #print target_ew_np
            true_list = np.append(true_list, target_ew_np)
            #print true_list
            _, pred_label_value = torch.max(output, 1)
            #print(pred_label_value.size())
            pred_list = np.append(pred_list, pred_label_value.cpu().numpy())
            #print(pred_list)
#            print("the length of pred list is {pred}".format(pred = len(pred_list)))
            # measure accuracy and record loss
            prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top3.update(prec3.item(), input.size(0))
            
                        

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10  == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top3=top3))



    # TENSORBOARD LOGGING
    # 1. Log scalar values (scalar summary)
    info = { 'loss': losses.avg, 'accuracy': top1.avg }

    for tag, value in info.items():
        logger_val.scalar_summary(tag, value, epoch)

    print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
          .format(top1=top1, top3=top3))
   # accuracy score, confusion_matrix and classification report from sklearn
    print('Confusion matrix: ')
    print confusion_matrix(true_list, pred_list)
    print('Accuracy score: ', accuracy_score(true_list, pred_list))
    print(classification_report(true_list,pred_list))

    return top1.avg


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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, csv_file, root_dir, sequence_length, indices=None, num_samples=None):
        #print('here')        
        self.root_dir = root_dir
        self.transform = transform
        self.landmarks = pd.read_csv(csv_file)
        self.sequence_length = sequence_length


        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)
        #print('here111')
        
        
    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is torchvision.datasets.MNIST:
            return dataset.train_labels[idx].item()
        elif dataset_type is torchvision.datasets.ImageFolder:
            return dataset.imgs[idx][1]
        else:
            #img_name_A = os.path.join('/scratch/shubodh/places365/rapyuta4_classes/train_all/', self.landmarks.iloc[idx, 0])
            label = self.landmarks.iloc[idx, 1]
            #print(img_name_A, label)
            label_all = self.landmarks.iloc[:,1]
            label_all_np = np.array(label_all)
            new_len = len(self.landmarks) - self.sequence_length + 1
           
            label_all_seq = np.zeros((new_len,20))

            for i in range(new_len):
                label_all_seq[i] = label_all_np[i:i+20]
                
            #freq_label = np.array(mode(label_all_seq.T))[0,:]
            #freq_label_final = freq_label.T.flatten()
            
            freq_label = label_all_seq[:,9]
            freq_label_final = freq_label.flatten()
            print "label{}".format(freq_label_final[idx])
            print "idx{}".format(idx)
            return (freq_label_final[idx])
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

class GetDataset(Dataset):
    def __init__(self, csv_file, root_dir, sequence_length, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.landmarks = pd.read_csv(csv_file)
        self.sequence_length = sequence_length
        self.inpa = np.load(root_dir + 'train_input_feature_21200x20x512_satyajit_model.npy')

    def __len__(self):
        return (len(self.landmarks) - self.sequence_length + 1)

    def __getitem__(self, idx):
        #img_name_A = os.path.join(self.root_dir, self.landmarks.iloc[idx, 0])
        label = self.landmarks.iloc[idx, 1] 
        label_all = self.landmarks.iloc[:,1]
        label_all_np = np.array(label_all)
        #print label_all_np.shape
        new_len = len(self.landmarks) - self.sequence_length + 1
       
        label_all_seq = np.zeros((new_len,20))

        for i in range(new_len):
            label_all_seq[i] = label_all_np[i:i+20]

        return (self.inpa[idx], label_all_seq[idx])


class GetTestDataset(Dataset):
    def __init__(self, csv_file, root_dir, sequence_length, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.landmarks = pd.read_csv(csv_file)
        self.sequence_length = sequence_length
        self.inpa = np.load(root_dir + 'test_input_feature_8400x20x512_satyajit_model.npy')
        self.rand = np.zeros([8419,512])

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, idx):
       # img_name_A = os.path.join(self.root_dir, self.landmarks.iloc[idx, 0])
        label = self.landmarks.iloc[idx, 1] 

        return (self.rand[idx], label)

class GetTempTestDataset(Dataset):
    def __init__(self, csv_file, root_dir, sequence_length, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.landmarks = pd.read_csv(csv_file)
        self.sequence_length = sequence_length
        self.inpa = np.load(root_dir + 'test_input_feature_8400x20x512_satyajit_model.npy')

    def __len__(self):
        return (len(self.landmarks) - self.sequence_length + 1)

    def __getitem__(self, idx):
       # img_name_A = os.path.join(self.root_dir, self.landmarks.iloc[idx, 0])
        label = self.landmarks.iloc[idx, 1] 
        label_all = self.landmarks.iloc[:,1]
        label_all_np = np.array(label_all)
        #print label_all_np.shape
        new_len = len(self.landmarks) - self.sequence_length + 1
       
        label_all_seq = np.zeros((new_len,20))

        for i in range(new_len):
            label_all_seq[i] = label_all_np[i:i+20]

        #print "labels shape {}, input shape{}".format(label_all_seq.shape,self.inpa.shape)
        #print "labels indi shape {}, input shape{}".format(label_all_seq[idx].shape,self.inpa[idx].shape)
        #print "labels {}, input {}".format(label_all_seq[idx],self.inpa[idx])
        #print(self.inpa[idx], label)
        #print(self.landmarks.iloc[idx, 0], label)
        #time.sleep(1)
        #print "idx {} name {}".format(idx,self.landmarks.iloc[idx, 0])
        return (self.inpa[idx], label_all_seq[idx])

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_all = lr_all * (0.1 ** ((epoch - 117) // 30))
#        for param_group in optimizer.param_groups:
#            param_group['lr'] = lr



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename + '_latest.pth.tar')
    if is_best:
        shutil.copyfile(filename + '_latest.pth.tar', filename + '_best.pth.tar')
 
def calculateTotalLoss(targ, preda):
   
    w = torch.Tensor([0.2,0.6,1.8,0.6]).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=w)
    return criterion(preda,targ)
    
    


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


#write object for tensorboard.
#writer = SummaryWriter('/home/tourani/Desktop/region-classification-matterport-pytorch/lf')

#MPObj = torch.load('/scratch/satyajittourani/saved_models_raputa_resize/latest_saved_state_005.pt').cuda()
if __name__ == '__main__':
    main()
