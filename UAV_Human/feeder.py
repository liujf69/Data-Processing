import torch
import numpy as np
from torch.utils.data import Dataset

class Feeder(Dataset):
    def __init__(self, data_path: str, data_split: str):
        super(Feeder, self).__init__()
        self.data_path = data_path
        self.data_split = data_split
        self.load_data()
        
    def load_data(self):
        npz_data = np.load(self.data_path)
        if self.data_split == 'train':
            self.data = npz_data['x_train']
            self.label = npz_data['y_train']
        else:
            assert self.data_split == 'test'
            self.data = npz_data['x_test']
            self.label = npz_data['y_test']
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        pose_data = self.data[idx]
        label = self.label[idx]
        
        return pose_data, label
    
if __name__ == "__main__":
    # Debug
    train_loader = torch.utils.data.DataLoader(
                dataset = Feeder(data_path = './pose_data/V1.npz', data_split = 'train'),
                batch_size = 4,
                shuffle = True,
                num_workers = 2,
                drop_last = False)
    
    val_loader = torch.utils.data.DataLoader(
            dataset = Feeder(data_path = './pose_data/V1.npz', data_split = 'test'),
            batch_size = 4,
            shuffle = False,
            num_workers = 2,
            drop_last = False)
    
    for batch_size, (data, label) in enumerate(train_loader):
        data = data.float() # B T M V C
        label = label.long() # B 1
        print("pasue")
    
            
        
    