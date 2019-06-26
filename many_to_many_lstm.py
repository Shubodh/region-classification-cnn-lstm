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
        self.hidden_size = hidden_size #128
        self.num_layers = num_layers #1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True) #basically weight matrix of dim 28,128,1
        self.fc = nn.Linear(hidden_size, num_classes) # basically weight matrix of dim (hid.. x num_classes) = (128,10)  
    
    def forward(self, x):
        # Set initial hidden and cell states 
        # Just ignore other dims, h0 and c0 are a 1D vector of size hidden_size
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) #x.size(0) is batch size
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        #print ('x dimensions: {}; h dimensions: {}; c dimensions: {}'.format(x.shape, h0.shape, c0.shape))
        # x dimensions: torch.Size([100, 28, 28]); h dimensions: torch.Size([1, 100, 128]); c dimensions: torch.Size([1, 100, 128])

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (100,28,128) = (batch_size, seq_length, hidden_size)
        # x: input of shape (batch, seq_len, input_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, :, :]) # out: shape (100,10) : multiply weight (128,10) with (100,28,128) 
        #print "out vector {} {}".format(out[1,:], out[1,27].shape)
        return out
#        for i in range(x.size(0)):
#            out[i] = self.fc(out[:, :, :])
#
#        
#        return out

model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)

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
        
        #print labels.shape
        
        labels = labels.repeat(10,1)
        labels = labels.permute(1,0)
        #print labels.shape

        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
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
        _, predicted = torch.max(outputs[:,-1].data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) 

# Save the model checkpoint
torch.save(model.state_dict(), 'model_many_to_many_lstm.ckpt')
