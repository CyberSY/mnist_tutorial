# -*- coding: utf-8 -*-
"""
Created on Wed May 12 16:09:42 2021

@author: Administrator
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch import optim
from tqdm import tqdm
import time


class SimpleNet(nn.Module):
    def __init__(self,num_classes = 10):
        super(SimpleNet,self).__init__()
        self.L1 = nn.Linear(784,784)
        self.L2 = nn.Linear(784,256)
        self.L3 = nn.Linear(256,10)
    
    def forward(self,x):
        out = F.elu(self.L1(x))
        out = F.elu(self.L2(out))
        out = F.elu(self.L3(out))
        out = F.softmax(out,dim = 1)
        return out
        
def evaluate(model, loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        x = x.view(128,784)
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()
    return correct / total

BATCH_SIZE = 128
NUM_EPOCHS = 10

# preprocessing
normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])

# download and load the data
train_dataset = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=transform, download=False)

# encapsulate them into dataloader form
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-3)
global_step = 0
for epoch in range(NUM_EPOCHS):
    for images, labels in tqdm(train_loader):
        images = images.view(BATCH_SIZE,784)
        images,labels = images.to(device), labels.to(device)
        model.train()
        logits = model(images)
        loss = criterion(logits,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_acc = evaluate(model, train_loader)
    print('train_acc:', train_acc)
    test_acc = evaluate(model, test_loader)
    print('test_acc:', test_acc)