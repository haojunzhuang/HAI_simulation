from torch.utils.data import Dataset

class MatrixDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X.float()
        self.Y = Y.float()
    
    def __len__(self):
        assert self.X.shape[0] == self.Y.shape[0]
        return self.X.shape[0]

    def __getitem__(self, index):
        # return self.X[index], self.Y[index]
        return {'data': self.X[index], 'recon': self.Y[index]}