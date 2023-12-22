
import os
import numpy as np
import random
from scipy import interpolate



def Normalization(InputVector):
    '''
    Normalization:
        X = ( X - Xmin) / ( Xmax - Xmin )
    '''
    OutputVector = (InputVector-InputVector.min()) / (InputVector.max() - InputVector.min())
    return OutputVector

def Linear1dInterpolate(Input_x, Input_y):
    '''
    scale spectrum intersection of sample to (5° to 24°) with 0.01° interval,
    the output is [1,1901], which is the input data for deep learning.
    '''
    F = interpolate.interp1d(Input_x, Input_y, kind='linear')
    NewIndex = np.linspace(5, 24, 1901)
    UpdatedVector = F(NewIndex)
    return UpdatedVector


###### ----- prepare theoretical training data----- #######

load_test = '/data/theoretical'
save_test = '/XRD/train'
index = 0
for ff in os.listdir(load_test):
    if ff.endswith('_y.npy'):
        print(ff)
        if ff.startswith('bast_'):
            label = [1,0,0]
            data_ff = os.path.join(load_test,ff)
            data_L = np.load(data_ff)
            for i in range(data_L.shape[0]):
                data_S = data_L[i,200:2101]
                save_data = os.path.join(save_test,'data',str(index)+'.npy')
                np.save(save_data,data_S)
                save_label = os.path.join(save_test,'label',str(index)+'.npy')
                np.save(save_label,label)                
                index = index + 1
       
        if ff.startswith('cal_'):
            label = [0,1,0]
            data_ff = os.path.join(load_test,ff)
            data_L = np.load(data_ff)
            for i in range(data_L.shape[0]):
                data_S = data_L[i,200:2101]
                save_data = os.path.join(save_test,'data',str(index)+'.npy')
                np.save(save_data,data_S)
                save_label = os.path.join(save_test,'label',str(index)+'.npy')
                np.save(save_label,label)                
                index = index + 1

        if ff.startswith('re_'):
            label = [0,0,1]
            data_ff = os.path.join(load_test,ff)
            data_L = np.load(data_ff)
            for i in range(data_L.shape[0]):
                data_S = data_L[i,200:2101]
                save_data = os.path.join(save_test,'data',str(index)+'.npy')
                np.save(save_data,data_S)
                save_label = os.path.join(save_test,'label',str(index)+'.npy')
                np.save(save_label,label)
                index = index + 1


####### -----prepare test data------  ########  
load_test = '/data/experimental/unmasked'
save_test = '/XRD/test_unmasked'

index = 0
for ff in os.listdir(load_test):
    if ff.startswith('xy_'):
        data_ff = os.path.join(load_test,ff)
        data_L = np.load(data_ff)
        new_data = Linear1dInterpolate(data_L[0,:],data_L[1,:])
    
        if '_bc_' in ff:
            label = [1,1,0]
        if '_c_' in ff:
            label = [0,1,0]
        if '_r_' in ff:
            label = [0,0,1]
        if '_n_' in ff:
            label = [0,0,0]
        
        save_data = os.path.join(save_test,'data',str(index)+'.npy')
        np.save(save_data,new_data)
        save_label = os.path.join(save_test,'label',str(index)+'.npy')
        np.save(save_label,label)
        index = index + 1

####### prepare test data  ########  
load_test = '/data/experimental/masked'
save_test = '/XRD/test_masked'

index = 0
for ff in os.listdir(load_test):
    if ff.startswith('xy_'):
        data_ff = os.path.join(load_test,ff)
        data_L = np.load(data_ff)
        new_data = Linear1dInterpolate(data_L[0,:],data_L[1,:])
    
        if '_bc_' in ff:
            label = [1,1,0]
        if '_c_' in ff:
            label = [0,1,0]
        if '_r_' in ff:
            label = [0,0,1]
        if '_n_' in ff:
            label = [0,0,0]
        
        save_data = os.path.join(save_test,'data',str(index)+'.npy')
        np.save(save_data,new_data)
        save_label = os.path.join(save_test,'label',str(index)+'.npy')
        np.save(save_label,label)
        index = index + 1
