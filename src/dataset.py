from torch.utils.data import Dataset, DataLoader
import torch

class BuckeyeDataset(Dataset):

    def __init__(self, wavs, labels, bounds):
        self.wavs= torch.tensor(wavs)
        self.labels= torch.tensor(labels)
        self.bounds= torch.tensor(bounds)

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        return self.wavs[idx], self.labels[idx], self.bounds[idx]
    
def get_loader(wavs, labels, bounds, batch_size, type='train'):
    shuffle=True
    if type!='train':
        shuffle=False
    dataset= BuckeyeDataset(wavs, labels, bounds)
    loader= DataLoader(dataset, batch_size= batch_size, shuffle= shuffle)
    return loader
    
