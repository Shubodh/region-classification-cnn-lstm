import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


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

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

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
