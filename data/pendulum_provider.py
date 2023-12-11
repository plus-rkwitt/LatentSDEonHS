"""Dataset provider for the Pendulm regression and interpolation task from

    Zeng S., Graf F. and Kwitt, R.
    Latent SDEs on Homogeneous Spaces
    NeurIPS 2023

    Data loading code is adapted (in parts) from
    
    https://github.com/boschresearch/Continuous-Recurrent-Units
    
    and
    
    https://github.com/ALRhub/rkn_share
"""

import os
import torch
import logging
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from .pendulum_generation import Pendulum
from .dataset_provider import DatasetProvider


class PendulumBase(Dataset):
    """Pendulum Dataset base class"""
    def __init__(self, data_dir, task, mode, impute_rate=0.5, sample_rate=0.5, random_state=0):
        
        assert task in ['regression', 'interpolation']
        assert mode in ['train', 'test', 'valid']
        self.mode = mode
        self.task = task
        data_dir = os.path.join(data_dir, 'pendulum')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        if task == 'regression':
            data_path = os.path.join(data_dir, f'pendulum_{task}.npz')
            if not os.path.exists(data_path):
                    logging.debug(f'Generating pendulum {mode} trajectories and saving to {data_path} ...')
                    generate_pendulums(data_path, task=task)
            data = dict(np.load(data_path))
            logging.debug(f'Loaded {mode:5s} data from {data_path}!')
            subsampled_data = subsample(
            data, sample_rate=sample_rate, random_state=random_state)
            train_obs, train_targets, test_obs, test_targets, validation_obs, validation_targets, train_time_points, test_time_points, validation_time_points = [torch.from_numpy(x) for x in subsampled_data]

        if task == 'interpolation':
            data_path = os.path.join(data_dir, f'pendulum_{task}_ir{impute_rate}.npz')
            if not os.path.exists(data_path):
                    logging.debug(f'Generating pendulum {mode} trajectories and saving to {data_path} ...')
                    generate_pendulums(data_path, task=task)
            data = dict(np.load(data_path))
            logging.debug(f'Loaded {mode:5s} data from {data_path}!')
            subsampled_data = subsample(data, sample_rate=sample_rate, imagepred=True, random_state=random_state)
            train_obs, train_targets, train_time_points, train_obs_valid, \
            test_obs, test_targets, test_time_points, test_obs_valid, \
            validation_obs, validation_targets, validation_time_points, validation_obs_valid = [torch.from_numpy(x) for x in subsampled_data]

        if mode == 'train':
            self.obs = train_obs
            self.tgt = train_targets
            self.tps = train_time_points
            self.msk = train_obs_valid.squeeze() if task == 'interpolation' else torch.ones_like(self.tps, dtype=bool)
        elif mode == 'test':
            self.obs = test_obs
            self.tgt = test_targets
            self.tps = test_time_points
            self.msk = test_obs_valid.squeeze() if task == 'interpolation' else torch.ones_like(self.tps, dtype=bool)
        elif mode == 'valid':
            self.obs = validation_obs
            self.tgt = validation_targets
            self.tps = validation_time_points
            self.msk = validation_obs_valid.squeeze() if task == 'interpolation' else torch.ones_like(self.tps, dtype=bool)

        self.obs = torch.permute(self.obs, [0, 1, 4, 2, 3])/255.0
        self.obs = self.obs.contiguous()  
        self.tps = self.tps.long()  
        self.tgt = self.tgt.float()    

        if task == 'interpolation':
            self.tgt = torch.permute(self.tgt, [0, 1, 4, 2, 3])/255.0
            self.tgt = self.tgt.contiguous()

    def __len__(self):
        return self.obs.shape[0]


class PendulumDataset(PendulumBase):
    
    num_timepoints = 100
    
    def __init__(self, data_dir, task, mode, impute_rate=0.5, sample_rate=0.5, random_state=0):
        super().__init__(data_dir, task, mode, impute_rate, sample_rate, random_state)      
        self._rewrite()  
        
    def _rewrite(self):
        """Rewrites dataset so that it can be input to mTAN."""

        # rewrite time points
        tps_new = torch.zeros_like(self.tps)
        for i in range(self.tps.shape[0]):    
            valid_tps = self.tps[i][self.msk[i]]
            tps_new[i,0:len(valid_tps)] = valid_tps

        # rewrite observations
        obs_new = torch.zeros_like(self.obs)
        for i in range(self.obs.shape[0]):
            valid_tps = self.tps[i][self.msk[i]]
            obs_new[i,0:len(valid_tps)] = self.obs[i,self.msk[i]]

        # rewrite mask
        obs_msk = torch.zeros_like(self.obs[:,:,0:1,0,0], dtype=torch.long)
        for i in range(self.obs.shape[0]):
            valid_tps = self.tps[i][self.msk[i]]
            obs_msk[i,0:len(valid_tps)] = 1
        obs_msk = obs_msk.unflatten(-1,(1,1,1)).expand_as(self.obs)

        self.inp_obs = obs_new
        self.inp_msk = obs_msk
        self.inp_tid = tps_new

        self.evd_msk = torch.ones_like(self.inp_msk)
        self.evd_tid = self.tps

        if self.task == 'regression':
            self.evd_obs = self.inp_obs
            self.aux_obs = self.tgt
            self.aux_msk = torch.ones_like(self.aux_obs, dtype=torch.long)
            self.aux_tid = self.evd_tid
        elif self.task == 'interpolation':
            self.evd_obs = self.tgt

    @property    
    def has_aux(self):
        return self.task == 'regression'
        
    def __getitem__(self, idx):
        inp_and_evd = {
            'inp_obs' : self.inp_obs[idx],
            'inp_msk' : self.inp_msk[idx],
            'inp_tid' : self.inp_tid[idx],
            'inp_tps' : self.inp_tid[idx]/self.num_timepoints,
            'evd_obs' : self.evd_obs[idx],
            'evd_msk' : self.evd_msk[idx],
            'evd_tid' : self.evd_tid[idx]
            }
        if self.has_aux:
            return {**inp_and_evd,
                'aux_obs' : self.aux_obs[idx],
                'aux_tid' : self.aux_tid[idx],
            }
        else:
            return inp_and_evd
            

class PendulumProvider(DatasetProvider):
    def __init__(self, data_dir: str=None, task:str=None):
        DatasetProvider.__init__(self)    
    
        assert task in ['regression', 'interpolation'], f'task {task} not supported'
        
        self.task = task
        self._ds_trn = PendulumDataset(data_dir, task, 'train')
        self._ds_tst = PendulumDataset(data_dir, task, 'test')
        self._ds_val = PendulumDataset(data_dir, task, 'valid')
    
    @property    
    def num_timepoints(self) -> int:
        return PendulumDataset.num_timepoints
    
    @property
    def num_train_samples(self) -> int:
        return len(self._ds_trn)
    
    @property 
    def num_test_samples(self) -> int:
        return len(self._ds_tst)
    
    @property
    def num_val_samples(self) -> int:
        return len(self._ds_val)
    
    def get_train_loader(self, **kwargs) -> DataLoader:
        return DataLoader(self._ds_trn, **kwargs)
    
    def get_test_loader(self, **kwargs) -> DataLoader:
        return DataLoader(self._ds_tst, **kwargs)
    
    def get_val_loader(self, **kwargs) -> DataLoader:
        return DataLoader(self._ds_val, **kwargs)
    

def subsample(data, sample_rate, imagepred=False, random_state=0):
    train_obs, train_targets, test_obs, test_targets, validation_obs, validation_targets = data["train_obs"], \
        data["train_targets"], data["test_obs"], data["test_targets"], data["validation_obs"], data["validation_targets"]
    seq_length = train_obs.shape[1]
    train_time_points = []
    test_time_points = []
    validation_time_points = []
    
    n = int(sample_rate*seq_length)

    if imagepred:
        train_obs_valid = data["train_obs_valid"]
        test_obs_valid = data["test_obs_valid"]
        validation_obs_valid = data["validation_obs_valid"]
        data_components = train_obs, train_targets, test_obs, test_targets, validation_obs, validation_targets, train_obs_valid, test_obs_valid, validation_obs_valid
        train_obs_sub, train_targets_sub, test_obs_sub, test_targets_sub, validation_obs_sub, validation_targets_sub, train_obs_valid_sub, test_obs_valid_sub, validation_obs_valid_sub = [
            np.zeros_like(x[:, :n, ...]) for x in data_components]
    else:
        data_components = train_obs, train_targets, test_obs, test_targets, validation_obs, validation_targets
        train_obs_sub, train_targets_sub, test_obs_sub, test_targets_sub, validation_obs_sub, validation_targets_sub = [
            np.zeros_like(x[:, :n, ...]) for x in data_components]

    for i in range(train_obs.shape[0]):
        rng_train = np.random.default_rng(random_state+i+train_obs.shape[0])
        choice = np.sort(rng_train.choice(seq_length, n, replace=False))
        train_time_points.append(choice)
        train_obs_sub[i, ...], train_targets_sub[i, ...] = [
            x[i, choice, ...] for x in [train_obs, train_targets]]
        if imagepred:
            train_obs_valid_sub[i, ...] = train_obs_valid[i, choice, ...]

    for i in range(test_obs.shape[0]):
        rng_test = np.random.default_rng(random_state+i)
        choice = np.sort(rng_test.choice(seq_length, n, replace=False))
        test_time_points.append(choice)
        test_obs_sub[i, ...], test_targets_sub[i, ...] = [
            x[i, choice, ...] for x in [test_obs, test_targets]]
        if imagepred:
            test_obs_valid_sub[i, ...] = test_obs_valid[i, choice, ...]
                        
    for i in range(validation_obs.shape[0]):
        rng_test = np.random.default_rng(random_state+i)
        choice = np.sort(rng_test.choice(seq_length, n, replace=False))
        validation_time_points.append(choice)
        validation_obs_sub[i, ...], validation_targets_sub[i, ...] = [
            x[i, choice, ...] for x in [validation_obs, validation_targets]]
        if imagepred:
            validation_obs_valid_sub[i, ...] = validation_obs_valid[i, choice, ...]        
            
    train_time_points, test_time_points, validation_time_points = \
        np.stack(train_time_points, 0), np.stack(test_time_points, 0), np.stack(validation_time_points, 0)

    if imagepred:
        return train_obs_sub, \
            train_targets_sub, \
            train_time_points, \
            train_obs_valid_sub, \
            test_obs_sub, \
            test_targets_sub, \
            test_time_points, \
            test_obs_valid_sub, \
            validation_obs_sub, \
            validation_targets_sub, \
            validation_time_points, \
            validation_obs_valid_sub, 
    else:
        return train_obs_sub, \
            train_targets_sub, \
            test_obs_sub, \
            test_targets_sub, \
            validation_obs_sub, \
            validation_targets_sub, \
            train_time_points, \
            test_time_points, \
            validation_time_points


"""The following code is aken from 

    https://github.com/ALRhub/rkn_share/ 

    and modified to generate an additional validation set.
"""
def generate_pendulums(file_path: str, task: str, impute_rate: float = 0.5):
    
    if task == 'interpolation':
        pend_params = Pendulum.pendulum_default_params()
        pend_params[Pendulum.FRICTION_KEY] = 0.1
        n = 100

        pendulum = Pendulum(24, observation_mode=Pendulum.OBSERVATION_MODE_LINE,
                            transition_noise_std=0.1, observation_noise_std=1e-5,
                            seed=42, pendulum_params=pend_params)
        rng = pendulum.random

        train_obs, _, _, _, train_ts = pendulum.sample_data_set(
            2000, n, full_targets=False)
        train_obs = np.expand_dims(train_obs, -1)
        train_targets = train_obs.copy()
        train_obs_valid = rng.rand(
            train_obs.shape[0], train_obs.shape[1], 1) > impute_rate
        train_obs_valid[:, :5] = True
        train_obs[np.logical_not(np.squeeze(train_obs_valid))] = 0

        test_obs, _, _, _, test_ts = pendulum.sample_data_set(
            1000, n, full_targets=False)
        test_obs = np.expand_dims(test_obs, -1)
        test_targets = test_obs.copy()
        test_obs_valid = rng.rand(
            test_obs.shape[0], test_obs.shape[1], 1) > impute_rate
        test_obs_valid[:, :5] = True
        test_obs[np.logical_not(np.squeeze(test_obs_valid))] = 0
        
        validation_obs, _, _, _, validation_ts = pendulum.sample_data_set(
            1000, n, full_targets=False)
        validation_obs = np.expand_dims(validation_obs, -1)
        validation_targets = validation_obs.copy()
        validation_obs_valid = rng.rand(
            validation_obs.shape[0], validation_obs.shape[1], 1) > impute_rate
        validation_obs_valid[:, :5] = True
        validation_obs[np.logical_not(np.squeeze(validation_obs_valid))] = 0
        
        np.savez_compressed(file_path,
                            train_obs=train_obs, 
                            train_targets=train_targets, 
                            train_obs_valid=train_obs_valid, 
                            train_ts=train_ts,
                            test_obs=test_obs, 
                            test_targets=test_targets, 
                            test_obs_valid=test_obs_valid, 
                            test_ts=test_ts,
                            validation_obs=validation_obs,
                            validation_targets=validation_targets,
                            validation_obs_valid=validation_obs_valid, 
                            validation_ts=validation_ts)
    
    elif task == 'regression':
        pend_params = Pendulum.pendulum_default_params()
        pend_params[Pendulum.FRICTION_KEY] = 0.1
        pend_params[Pendulum.DT_KEY] = 0.01
        n = 100

        pendulum = Pendulum(24, observation_mode=Pendulum.OBSERVATION_MODE_LINE,
                            transition_noise_std=0.1, observation_noise_std=1e-5,
                            seed=42, pendulum_params=pend_params)

        train_obs, train_targets, _, _, train_ts = pendulum.sample_data_set(
            2000, n, full_targets=False)
        train_obs, _ = pendulum.add_observation_noise(train_obs, first_n_clean=5, r=0.2, t_ll=0.0, t_lu=0.25, t_ul=0.75,
                                                      t_uu=1.0)
        train_obs = np.expand_dims(train_obs, -1)

        test_obs, test_targets, _, _, test_ts = pendulum.sample_data_set(
            1000, n, full_targets=False)
        test_obs, _ = pendulum.add_observation_noise(test_obs, first_n_clean=5, r=0.2, t_ll=0.0, t_lu=0.25, t_ul=0.75,
                                                     t_uu=1.0)
        test_obs = np.expand_dims(test_obs, -1)
        
        validation_obs, validation_targets, _, _, validation_ts = pendulum.sample_data_set(
            1000, n, full_targets=False)
        validation_obs, _ = pendulum.add_observation_noise(validation_obs, first_n_clean=5, r=0.2, t_ll=0.0, t_lu=0.25, t_ul=0.75,
                                                           t_uu=1.0)
        validation_obs = np.expand_dims(validation_obs, -1)
        
        np.savez_compressed(file_path,
                            train_obs=train_obs, 
                            train_targets=train_targets, 
                            train_ts=train_ts,
                            test_obs=test_obs, 
                            test_targets=test_targets, 
                            test_ts=test_ts,
                            validation_obs=validation_obs, 
                            validation_targets=validation_targets, 
                            validation_ts=validation_ts)