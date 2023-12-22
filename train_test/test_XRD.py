# 

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch import optim
import torch.nn as nn
import torch

from DataLoader import Dataset_XRD
from model_XRD import *

# Please modify the data_path by yourself
data_path = '/XRD/dataset'
test_dataset = Dataset_XRD(data_path, 'test')
print('Test data num : ', len(test_dataset))

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=60, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Network_XRD()
## load saved_model for test
model.load_state_dict(torch.load('saved_model.pth'))
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
        