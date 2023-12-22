# MLXRD

This is the opensource code for the PNNL LDRD project MDCRUST.  In this code, Wwe apply a multi-task neural network to identify multi-compounds (Bastnaesite, Calcite, Re) from XRD (X-ray diffraction) data collected from a hydrothermal fluid system in synchrotron. 

To run the code, Python3 and PyTorch are needed.

## Folders:
### dataPrepare:
- Data is the raw data, including theoretical and experiment dataset. Theoretical data is used for training, and experiment data is used for testing. 
- DataPrepare.py is used to prepare dataset for training and testing from raw data.

### train_test:
- DataLoader.py: load and preprocess the data.
- model_XRD.py: define neural network model.
- train_XRD: train model with training dataset and test.
- test_XRD: test data in trained model.

# Run

## To prepare data:
```text
python DataPrepare.py
```text

## Train
```text
python train_XRD.py
```

## Test
```text
python test_XRD.py
```

## Contributors: 
Yanfei Li, Xiaodong Zhao, Juejing Liu, Tong Geng, Ang Li, Xin Zhang. 


## PNNL IPID-32938, Export Control: EAR99
