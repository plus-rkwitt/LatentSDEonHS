"""Dataset provider for toy datasets (e.g., irregular sine)."""

import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .dataset_provider import DatasetProvider

class IrregularSineDataset(Dataset):    
    
    num_timepoints = 101
    
    def __init__(self, num_samples:int = 1, lower_limit:float = 0.2, upper_limit: float = 0.8, mode:str = 'train'):

        self.mode = mode
        self.num_samples = num_samples
        
        lower_limit = int(lower_limit*(IrregularSineDataset.num_timepoints-1))
        upper_limit = int(upper_limit*(IrregularSineDataset.num_timepoints-1))
        
        self.evd_tid = torch.stack([torch.randperm(upper_limit-lower_limit)[0:16].sort().values+lower_limit for _ in range(self.num_samples)])
        self.evd_obs = (torch.sin(2*(self.evd_tid/(self.num_timepoints-1)) * (2. * torch.pi)) * 0.8).unsqueeze(-1)
        self.evd_msk = torch.ones_like(self.evd_obs)
        self.inp_tid = torch.zeros_like(self.evd_tid)
        self.inp_obs = torch.zeros_like(self.evd_obs)
        self.inp_msk = torch.zeros_like(self.evd_msk)
    
    def __getitem__(self, idx):
        inp_and_evd = {
            'inp_obs' : self.inp_obs[idx],
            'inp_msk' : self.inp_msk[idx],
            'inp_tid' : self.inp_tid[idx],
            'inp_tps' : self.inp_tid[idx],
            'evd_obs' : self.evd_obs[idx],
            'evd_msk' : self.evd_msk[idx],
            'evd_tid' : self.evd_tid[idx]
            }
        return inp_and_evd

    @property    
    def has_aux(self):
        return False

    def __len__(self):
        return self.num_samples
    
    
class IrregularSineProvider(DatasetProvider):
    def __init__(self, num_samples: int=1):
        DatasetProvider.__init__(self)    
        
        # possibly support testing/validation data  
        self._ds_trn = IrregularSineDataset(num_samples=num_samples, mode='train')
        
    @property    
    def num_timepoints(self):
        return IrregularSineDataset.num_timepoints
    
    @property
    def num_train_samples(self):
        return len(self._ds_trn)
    
    def get_train_loader(self, **kwargs):
        return DataLoader(self._ds_trn, **kwargs)