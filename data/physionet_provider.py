"""Dataset provider for the PhysioNet (2012) interpolation task.

Data loading code is taken and dadpated (in parts) from
    https://github.com/reml-lab/mTAN

Authors: Sebastian Zeng, Florian Graf, Roland Kwitt (2023)
"""

import os
import tarfile
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.utils import download_url
from torch.distributions import Categorical
from torch.utils.data.dataloader import default_collate

from sklearn.model_selection import train_test_split
from .common import get_data_min_max, variable_time_collate_fn, normalize_masked_data
from .dataset_provider import DatasetProvider

    

class PhysioNet(object):
    urls = [
        'https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download',
        'https://physionet.org/files/challenge-2012/1.0.0/set-b.tar.gz?download',
    ]

    outcome_urls = ['https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt']

    params = [
        'Age', 
        'Gender', 
        'Height', 
        'ICUType', 
        'Weight', 
        'Albumin', 
        'ALP', 
        'ALT', 
        'AST', 
        'Bilirubin', 
        'BUN',
        'Cholesterol', 
        'Creatinine', 
        'DiasABP', 
        'FiO2', 
        'GCS', 
        'Glucose', 
        'HCO3', 
        'HCT', 
        'HR', 
        'K', 
        'Lactate', 
        'Mg',
        'MAP', 
        'MechVent', 
        'Na', 
        'NIDiasABP', 
        'NIMAP', 
        'NISysABP', 
        'PaCO2', 
        'PaO2', 
        'pH', 
        'Platelets', 
        'RespRate',
        'SaO2', 
        'SysABP', 
        'Temp', 
        'TroponinI', 
        'TroponinT', 
        'Urine', 
        'WBC']

    params_dict = {k: i for i, k in enumerate(params)}

    labels = [ "SAPS-I", "SOFA", "Length_of_stay", "Survival", "In-hospital_death" ]
    labels_dict = {k: i for i, k in enumerate(labels)}

    def __init__(self, root, train=True, download=False, quantization = 0.1, n_samples = None):

        self.root = root
        self.train = train
        self.reduce = "average"
        self.quantization = quantization

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
    
        self.data = torch.load(os.path.join(self.processed_folder, data_file))
        self.labels = torch.load(os.path.join(self.processed_folder, self.label_file))

        if n_samples is not None:
            self.data = self.data[:n_samples]
            self.labels = self.labels[:n_samples]


    def download(self):
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # Download outcome data
        for url in self.outcome_urls:
            filename = url.rpartition('/')[2]
            download_url(url, self.raw_folder, filename, None)

            txtfile = os.path.join(self.raw_folder, filename)
            with open(txtfile) as f:
                lines = f.readlines()
                outcomes = {}
                for l in lines[1:]:
                    l = l.rstrip().split(',')
                    record_id, labels = l[0], np.array(l[1:]).astype(float)
                    outcomes[record_id] = torch.Tensor(labels)

            torch.save(labels, os.path.join(self.processed_folder, filename.split('.')[0] + '.pt'))

        for url in self.urls:
            filename = url.rpartition('/')[2]
            download_url(url, self.raw_folder, filename, None)
            tar = tarfile.open(os.path.join(self.raw_folder, filename), "r:gz")
            tar.extractall(self.raw_folder)
            tar.close()

            dirname = os.path.join(self.raw_folder, filename.split('.')[0])
            patients = []
            total = 0

            for txtfile in os.listdir(dirname):
                record_id = txtfile.split('.')[0]
                with open(os.path.join(dirname, txtfile)) as f:
                    lines = f.readlines()
                    prev_time = 0
                    tt = [0.]
                    vals = [torch.zeros(len(self.params))]
                    mask = [torch.zeros(len(self.params))]
                    nobs = [torch.zeros(len(self.params))]
                
                    for l in lines[1:]:
                        total += 1
                        time, param, val = l.split(',')
                        time = float(time.split(':')[0]) + float(time.split(':')[1]) / 60.
                        time = round(time / self.quantization) * self.quantization

                        if time != prev_time:
                            tt.append(time)
                            vals.append(torch.zeros(len(self.params)))
                            mask.append(torch.zeros(len(self.params)))
                            nobs.append(torch.zeros(len(self.params)))
                            prev_time = time

                        if param in self.params_dict:
                            n_observations = nobs[-1][self.params_dict[param]]
                            if self.reduce == 'average' and n_observations > 0:
                                prev_val = vals[-1][self.params_dict[param]]
                                new_val = (prev_val * n_observations + float(val)) / (n_observations + 1)
                                vals[-1][self.params_dict[param]] = new_val
                            else:
                                vals[-1][self.params_dict[param]] = float(val)
                                
                            mask[-1][self.params_dict[param]] = 1
                            nobs[-1][self.params_dict[param]] += 1
                        else:
                            assert param == 'RecordID', 'Read unexpected param {}'.format(param)
                
                tt = torch.tensor(tt)
                vals = torch.stack(vals)
                mask = torch.stack(mask)

                labels = None
                if record_id in outcomes:
                    labels = outcomes[record_id]
                    labels = labels[4] # mortality

                patients.append((record_id, tt, vals, mask, labels))

            torch.save(
                patients,
                os.path.join(self.processed_folder, 
                filename.split('.')[0] + "_" + str(self.quantization) + '.pt')
            )
                
    def _check_exists(self):
        for url in self.urls:
            filename = url.rpartition('/')[2]

            if not os.path.exists(
                os.path.join(self.processed_folder, 
                filename.split('.')[0] + "_" + str(self.quantization) + '.pt')
            ):
                return False
            return True

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def training_file(self):
        return 'set-a_{}.pt'.format(self.quantization)

    @property
    def test_file(self):
        return 'set-b_{}.pt'.format(self.quantization)

    @property
    def label_file(self):
        return 'Outcomes-a.pt'

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_label(self, record_id):
        return self.labels[record_id]


class PhysioNetDataset(Dataset):
    
    input_dim = None  # nr. of different measurements per time point
    
    def __init__(self, data_dir: str, mode: str='train', quantization: float=0.1, download: bool=False, random_state: int=42):
        
        self._quantization = quantization
        trn_obj = PhysioNet(data_dir, train=True, quantization=self._quantization, download=download, n_samples=8000)
        tst_obj = PhysioNet(data_dir, train=False, quantization=self._quantization, download=download, n_samples=8000)
        all_obj = trn_obj[:] + tst_obj[:]
        
        trn_data, tst_data = train_test_split(all_obj, train_size=0.8, random_state=random_state, shuffle=True)
        data_min, data_max = get_data_min_max(all_obj)
        _, _, vals, _, _ = all_obj[0]
        PhysioNetDataset.input_dim = vals.size(-1)

        len_tt = [ex[1].size(0) for ex in all_obj]
        maxlen = np.max(len_tt) # max. nr. of available timepoints at given quantization
        
        if mode=='train':
            data = trn_data
        elif mode=='test':
            data = tst_data 
            
        obs = torch.zeros([len(data), maxlen, PhysioNetDataset.input_dim])
        msk = torch.zeros([len(data), maxlen, PhysioNetDataset.input_dim])
        tps = torch.zeros([len(data), maxlen])

        for b, (_, record_tps, record_obs, record_msk,_) in enumerate(data):
            currlen = record_tps.size(0)
            obs[b, :currlen] = record_obs
            msk[b, :currlen] = record_msk
            tps[b, :currlen] = record_tps
        
        obs, _, _ = normalize_masked_data(obs, msk, data_min, data_max)
        
        tid = (tps/self._quantization).round().long()
        if torch.max(tps) != 0.:
            tps = tps / torch.max(tps)
        
        self.evd_obs = obs
        self.evd_msk = msk.long()
        self.evd_tid = tid.long()
        self.evd_tps = tps
        self.data_min = data_min
        self.data_max = data_max
        self.feature_names = PhysioNet.params
        
        self.num_timepoints = int(np.round(48./self._quantization))+1

    @property    
    def has_aux(self):
        return False

    def __len__(self):
        return len(self.evd_obs)

    def __getitem__(self, idx):
        inp_and_evd = {
            'evd_obs' : self.evd_obs[idx],
            'evd_msk' : self.evd_msk[idx],
            'evd_tid' : self.evd_tid[idx],
            'evd_tps' : self.evd_tps[idx]
            }
        return inp_and_evd


class PhysioNetProvider(DatasetProvider):
    def __init__(self, data_dir=None, quantization=0.1, sample_tp=0.5, random_state=42):
        DatasetProvider.__init__(self)
    
        self._sample_tp = sample_tp
        self._quantization = quantization
        self._ds_trn = PhysioNetDataset(data_dir, 'train', quantization=quantization, download=True, random_state=random_state)
        self._ds_tst = PhysioNetDataset(data_dir, 'test', quantization=quantization, download=True, random_state=random_state)
        
        assert self._ds_trn.num_timepoints == self._ds_tst.num_timepoints
        assert torch.all(self._ds_trn.data_min == self._ds_tst.data_min)
        assert torch.all(self._ds_trn.data_max == self._ds_tst.data_max)
        
    @property 
    def input_dim(self):
        return PhysioNetDataset.input_dim

    @property
    def sample_tp(self):
        return self._sample_tp

    @property
    def data_min(self):
        return self._ds_trn.data_min
    
    @property 
    def data_max(self):
        return self._ds_trn.data_max

    @property    
    def quantization(self):
        return self._quantization

    @property    
    def num_timepoints(self):
        return self._ds_trn.num_timepoints
    
    @property
    def num_train_samples(self):
        return len(self._ds_trn)
    
    @property 
    def num_test_samples(self):
        return len(self._ds_tst)
    
    @property 
    def num_val_samples(self):
        raise NotImplementedError
    
    def _collate(self, data):
        batch = default_collate(data)
        inp_obs, inp_msk, inp_tid = subsample_timepoints(
            batch['evd_obs'].clone(), 
            batch['evd_msk'].clone(),
            batch['evd_tid'].clone(), self.sample_tp)
        batch['inp_obs'] = inp_obs
        batch['inp_tps'] = inp_tid/(self.num_timepoints-1)
        batch['inp_msk'] = inp_msk
        batch['inp_tid'] = inp_tid 
        return batch
        
    def get_train_loader(self, **kwargs):
        return DataLoader(self._ds_trn, collate_fn=self._collate, **kwargs)
    
    def get_test_loader(self, **kwargs):
        return DataLoader(self._ds_tst, collate_fn=self._collate, **kwargs)
    
    
def subsample_timepoints(data, mask, tid, p=1.):
    assert 0. <= p <= 1.
    if p == 1.:
        sub_data, sub_mask, sub_tid = data, mask, tid
    else:
        tp_msk = torch.rand(tid.shape, device=tid.device) <= p # -> [batch_size, num_time_points] 
        sub_tid = tid * tp_msk
        tp_msk = tp_msk.unsqueeze(-1).expand_as(data)
        sub_data, sub_mask = (x * tp_msk for x in [data, mask])
    return sub_data, sub_mask, sub_tid