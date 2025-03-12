import torch
import xarray as xr

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, ds_datax, ds_datay):
        self.x_data = ds_datax
        self.y_data = ds_datay

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.x_data.shape[0]

def load_data():
    x = torch.load('data/xtrain_nl2.pt')
    y = torch.load('data/ytrain_nl2.pt')
    x_valid = torch.load('data/x_valid_nl2.pt')
    y_valid = torch.load('data/y_valid_nl2.pt')
    norm_data = xr.open_dataset('data/norm_data_nl2.nc')
    return x, y, x_valid, y_valid, norm_data

