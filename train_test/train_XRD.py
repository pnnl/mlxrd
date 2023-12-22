# for model training and testing

import os
import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch import optim
import torch.nn as nn
import torch

from DataLoader import Dataset_XRD
from model_XRD import *

# Please modify the data_path by yourself
data_path = '/XRD/dataset'

train_dataset = Dataset_XRD(data_path, 'train')
test_dataset = Dataset_XRD(data_path, 'test')
print('Train data num : ', len(train_dataset))
print('Test data num : ', len(test_dataset))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=60, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=60, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Network_XRD()

optimizer = optim.Adam(model.parameters(),lr=1e-5,weight_decay=1e-8)

#criterion = nn.CrossEntropyLoss(torch.tensor([1.0,1.0]))
criterion = nn.CrossEntropyLoss(torch.tensor([1.0,1.6]))

for epoch in range(300):

    model.train()
    for image, label in train_loader:
    
        optimizer.zero_grad()
        image = image.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.long)
        pred = model(image)
     
        loss_0 = criterion(pred[0], label[:,0])
        loss_1 = criterion(pred[1], label[:,1])
        loss_2 = criterion(pred[2], label[:,2])

        loss = loss_0 + loss_1 + loss_2

        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    print('epoch : ', epoch, ', training loss : ',train_loss/len(train_dataset))


    model.eval()

    for image, label in test_loader:
        image = image.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.long)
      
        pred = model(image)

        test_b += torch.sum(torch.argmax(pred[0], 1) == label[:,0]).cpu()
        test_c += torch.sum(torch.argmax(pred[1], 1) == label[:,1]).cpu()
        test_r += torch.sum(torch.argmax(pred[2], 1) == label[:,2]).cpu()


    test_bb.append(test_b / len(test_dataset))  
    test_cc.append(test_c / len(test_dataset))
    test_rr.append(test_r / len(test_dataset))


    print( 'test_b', test_b / len(test_dataset),', test c: ', test_c / len(test_dataset), ', test r: ', test_r / len(test_dataset))
        
torch.save(model.state_dict(), 'saved_model.pth')
