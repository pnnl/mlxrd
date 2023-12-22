from torch.utils.data import Dataset
from torch import optim
import torch.nn as nn
import torch
import numpy as np
import os

def Norm(InputVector):
    '''
    Normalization:
        X = ( X - Xmin) / ( Xmax - Xmin )
    '''
    Input_map = InputVector.max() - InputVector.min()
    OutputVector = (InputVector-InputVector.min()) / Input_map

    return OutputVector


class Dataset_XRD(Dataset):
    '''
    Dataloader for XRD dataset:
    DataPath: Path for the dataset
    DatasetClass: class of train or test.
    two files: data and label ([b,c,r]). 
               b is 1 or 0, meaning there is Bastnaesite or not, the same with c and r
               c is for Calcite and r is for Re.
    noise: Bool. Adding noise to the theoretical training data or not.
           for unmasked test data, add noise, for masked test data, do not add noise.
    '''
    def __init__(self, DataPath, DatasetClass):
        
        self.data_path = []
        self.label = []
        self.type = DatasetClass
        self.noise = True    #  set False for masked dataset   

        load_path = os.path.join(DataPath, DatasetClass)
        load_data = os.path.join(load_path,'data')
        load_label = os.path.join(load_path,'label')
        for s in os.listdir(load_data):
            if s.endswith('.npy'):
                da_path = os.path.join(load_data,s)
                self.data_path.append(da_path)
                la_path = os.path.join(load_label,s)
                t_label = np.load(la_path)
                self.label.append(t_label)
                               
    def __getitem__(self, index):
        da_load = self.data_path[index]
        data_raw = np.load(da_load)
        data_norm  = Norm(data_raw)
        if self.noise:
            #  add random noise to theoretical training data
            if 'train' in self.type:
                if not 'test_' in da_load:  
                    noise_add = np.random.normal(scale = 0.05, size = np.size(data_raw))
                    data_norm = data_norm + noise_add 
                    data_norm  = Norm(data_norm)
        
        data = torch.tensor(data_norm)
        data = data.reshape((1, data.shape[0]))   
        label_raw = self.label[index]
        label = torch.tensor(label_raw)
        return data, label

    def __len__(self):
        return len(self.label)