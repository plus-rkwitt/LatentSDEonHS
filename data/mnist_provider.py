"""Dataset provider for the RotatingMNIST interpolation task from 

    Zeng S., Graf F. and Kwitt, R.
    Latent SDEs on Homogeneous Spaces
    NeurIPS 2023

    Note: This is an adjusted version of the data loading code 
    from https://github.com/cagatayyildiz/ODE2VAE
"""

import os
import torch
import gdown
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from .dataset_provider import DatasetProvider
    

class RotatingMNISTSDataset(Dataset):    
    
    num_timepoints = 16
    
    def __init__(self, data_dir, mode='train', download=False, random_state=42):
    
        self.N = 360 
        self.mode = mode
        self.data_dir = data_dir
        self.random_state = random_state
        self.file_dir = os.path.join(self.data_dir, 'rot_mnist')
        self.url = 'https://drive.google.com/uc?id=1Ax5swK4YtilC8DCxSHXOngcEZ71bH7hw'
        if download:
            self._download()

        self._load_mnist_nonuniform_data(self.file_dir)    
    
    def _check_exists(self) -> bool:
        """Checks if rot-mnist-3s.mat file exists.

        Returns:
            bool: True if exists, else False
        """

    def _download(self) -> None:
        """Downloads rot-mnist-3s.mat file and establishes directory structure."""
        if self._check_exists():
            return
        os.makedirs(self.file_dir, exist_ok=True)
        gdown.download(self.url, os.path.join(self.file_dir, 'rot-mnist-3s.mat'), quiet=False)
        
    def _load_mnist_nonuniform_data(self, file_dir) -> None:
        """Loads and prepares time series of rotating 3's according to

        https://github.com/cagatayyildiz/ODE2VAE
        
        but splits data into training/validation/testing via scikit-learn.
        """
        data = sio.loadmat(os.path.join(file_dir, 'rot-mnist-3s.mat'))
        Xtr = np.squeeze(data['X'])
        Xtr = np.flip(Xtr,1)
        
        Xtr, Xtest = train_test_split(
            Xtr, 
            train_size=self.N+self.N//10, 
            test_size=self.N, 
            shuffle=True, 
            random_state=self.random_state)
        
        if self.mode == 'train':
            Xfull = Xtr
            [N,T,D] = Xtr.shape
            removed_angle = 3
            num_gaps = 5
            ttr = np.zeros([N,T-num_gaps])
            Xtr = np.zeros([N,T-num_gaps,D])
            for i in range(N):
                idx = np.arange(0,T)
                d = {removed_angle}
                while len(d) < num_gaps:
                    d.add(np.random.randint(1,T)) # keep 0-th image as initial data point
                idx = np.delete(idx,list(d))
                Xtr[i,:,:] = Xfull[i,idx,:]
                ttr[i,:] = idx                
            self.X = Xtr[:self.N]
            self.t = ttr[:self.N]
        elif self.mode == 'test':
            [N,T,D] = Xtest.shape
            self.X = Xtest
            self.t = np.tile(np.arange(0,T).reshape((1,-1)),[N,1])
        elif self.mode == 'valid':
            self.X = Xtr[self.N:] # remaining self.N//10
            [N,T,D] = self.X.shape
            self.t = np.tile(np.arange(0,T).reshape((1,-1)),[N,1])

        self.inp_obs = torch.tensor(self.X, dtype=torch.float32).view(self.X.shape[0:2] + torch.Size([1,28,28]))[:,0]
        self.inp_msk = torch.ones(self.X.shape[0:2] + torch.Size([1,28,28])).long()
        self.inp_tid = torch.tensor(self.t).long()
        self.evd_obs = torch.tensor(self.X, dtype=torch.float32).view(self.X.shape[0:2] + torch.Size([1,28,28]))
        self.evd_msk = torch.ones(self.X.shape[0:2] + torch.Size([1,28,28])).long()
        self.evd_tid = torch.tensor(self.t).long()
        self.input_dim = 28*28    
    
    @property    
    def has_aux(self):
        return False
    
    def __len__(self):
        return len(self.inp_obs)
    
    def __getitem__(self, idx):
        inp_and_evd = {
            'inp_obs' : self.inp_obs[idx],
            'inp_msk' : self.inp_msk[idx],
            'inp_tid' : self.inp_tid[idx],
            'inp_tps' : self.inp_tid[idx]/RotatingMNISTSDataset.num_timepoints,
            'evd_obs' : self.evd_obs[idx],
            'evd_msk' : self.evd_msk[idx],
            'evd_tid' : self.evd_tid[idx]
        }
        return inp_and_evd
    

class RotatingMNISTProvider(DatasetProvider):
    def __init__(self, data_dir=None, download=False, random_state=42):
        DatasetProvider.__init__(self)

        self._ds_trn = RotatingMNISTSDataset(data_dir, 'train', download=download, random_state=random_state)
        self._ds_tst = RotatingMNISTSDataset(data_dir, 'test', download=download, random_state=random_state)
        self._ds_val = RotatingMNISTSDataset(data_dir, 'valid', download=download, random_state=random_state)       
        
    @property    
    def num_timepoints(self):
        return RotatingMNISTSDataset.num_timepoints
    
    @property 
    def num_test_samples(self) -> int:
        return len(self._ds_tst)
    
    @property 
    def num_train_samples(self) -> int:
        return len(self._ds_trn)
    
    @property 
    def num_val_samples(self) -> int:
        return len(self._ds_val)
    
    def get_train_loader(self, **kwargs)  -> DataLoader:
        return DataLoader(self._ds_trn, **kwargs)    
        
    def get_test_loader(self, **kwargs) -> DataLoader:
        return DataLoader(self._ds_tst, **kwargs)
    
    def get_val_loader(self, **kwargs) -> DataLoader:
        return DataLoader(self._ds_val, **kwargs)

