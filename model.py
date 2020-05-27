import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import time

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20
# percentage of training set to use as validation
valid_size = 0.2# convert data to torch.FloatTensor
transform = transforms.ToTensor()# choose the training and testing datasets
train_data = datasets.MNIST(root = 'data', train = True, download = True, transform = transform)
test_data = datasets.MNIST(root = 'data', train = False, download = True, transform = transform)# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_index, valid_index = indices[split:], indices[:split]# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_index)
valid_sampler = SubsetRandomSampler(valid_index)# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size,
                                           sampler = train_sampler, num_workers = num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size,
                                          sampler = valid_sampler, num_workers = num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size,
                                         num_workers = num_workers)


import torch.nn as nn
import torch.nn.functional as F# define NN architecture
from GC import *

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # number of hidden nodes in each layer (512)
        hidden_1 = 512
        hidden_2 = 512
        # linear layer (784 -> hidden_1)
        self.fc1 = nn.Linear(28*28, 512)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(512,512)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(512,10)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.droput = nn.Dropout(0.2)

    def forward(self,x):
        # flatten image input
        x = x.view(-1,28*28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.droput(x)
         # add hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.droput(x)
        # add output layer
        x = self.fc3(x)
        return x# initialize the NN
model = Net()
print(model)


# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()# specify optimizer (stochastic gradient descent) and learning rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)


# number of epochs to train the model
n_epochs = 10# initialize tracker for minimum validation loss
valid_loss_min = np.Inf  # set initial "min" to infinity

train_loss_hist_1 = []
val_acc_hist_1 = []
start_time = time.time()

for epoch in range(n_epochs):
    # monitor losses
    train_loss = 0
    valid_loss = 0


    ###################
    # train the model #
    ###################
    model.train() # prep model for training
    for data,label in train_loader:
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output,label)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item() * data.size(0)

     ######################
    # validate the model #
    ######################

    model.eval()  # prep model for evaluation
    total = 0
    correct = 0
    for data,label in valid_loader:
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

        # calculate the loss
        loss = criterion(output,label)
        # update running validation loss
        valid_loss = loss.item() * data.size(0)

    # print training/validation statistics
    # calculate average loss over an epoch
    train_loss = train_loss / len(train_loader.sampler)
    valid_acc = correct/total

    train_loss_hist_1.append(train_loss)
    val_acc_hist_1.append(valid_acc)

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation acc: {:.6f}'.format(
        epoch+1,
        train_loss,
        valid_acc
        ))
    '''
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model.pt')
        valid_loss_min = valid_loss
    '''
time_1 = time.time() - start_time


transform = transforms.ToTensor()# choose the training and testing datasets
train_data = datasets.MNIST(root = 'data', train = True, download = True, transform = transform)
test_data = datasets.MNIST(root = 'data', train = False, download = True, transform = transform)# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_index, valid_index = indices[split:], indices[:split]# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_index)
valid_sampler = SubsetRandomSampler(valid_index)# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size,
                                           sampler = train_sampler, num_workers = num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size,
                                          sampler = valid_sampler, num_workers = num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size,
                                         num_workers = num_workers)

model = Net()
print(model)


# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()# specify optimizer (stochastic gradient descent) and learning rate = 0.01
optimizer = SGD_GC(model.parameters(), lr = 0.01)


# number of epochs to train the model
valid_loss_min = np.Inf  # set initial "min" to infinity

train_loss_hist_2 = []
val_acc_hist_2 = []
start_time = time.time()

for epoch in range(n_epochs):
    # monitor losses
    train_loss = 0
    valid_loss = 0


    ###################
    # train the model #
    ###################
    model.train() # prep model for training
    for data,label in train_loader:
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output,label)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item() * data.size(0)

     ######################
    # validate the model #
    ######################

    model.eval()  # prep model for evaluation
    total = 0
    correct = 0
    for data,label in valid_loader:
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

        # calculate the loss
        loss = criterion(output,label)
        # update running validation loss
        valid_loss = loss.item() * data.size(0)

    # print training/validation statistics
    # calculate average loss over an epoch
    train_loss = train_loss / len(train_loader.sampler)
    valid_acc = correct/total

    train_loss_hist_2.append(train_loss)
    val_acc_hist_2.append(valid_acc)

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation acc: {:.6f}'.format(
        epoch+1,
        train_loss,
        valid_acc
        ))
time_2 = time.time() - start_time

print("---Training time: %s seconds ---" % (time_1))
plt.plot(np.array(train_loss_hist_1), label = "SGC training loss")
plt.plot(np.array(train_loss_hist_2), label = "SGC with GD training loss")
plt.show()
plt.plot(np.array(val_acc_hist_1), label = "SGC test accuracy")
plt.plot(np.array(val_acc_hist_2), label = "SGC with GD test accuracy")
plt.show()
