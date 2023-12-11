"""Generic template for a dataset provider from

    Zeng S., Graf F. and Kwitt, R.
    Latent SDEs on Homogeneous Spaces
    NeurIPS 2023
"""

class DatasetProvider():
    def __init__(self):
        pass
        
    @property
    def data_min(self):
        raise NotImplementedError()
    
    @property
    def data_max(self):
        raise NotImplementedError()
    
    @property 
    def num_test_samples(self):
        raise NotImplementedError()
    
    @property 
    def num_train_samples(self):
        raise NotImplementedError()
    
    @property 
    def num_val_samples(self):
        raise NotImplementedError()
    
    def get_train_loader(self, **kwargs):
        raise NotImplementedError()
    
    def get_test_loader(self, **kwargs):
        raise NotImplementedError()
    
    def get_val_loader(self, **kwargs):
        raise NotImplementedError()
    
    def decomposer(self):
        raise NotImplementedError()
