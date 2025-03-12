# data/data_generator.py

import os
import numpy as np
import pandas as pd
import xarray as xr
from glob import glob
import torch
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
#  Load and Stack Datasets
# =============================================================================
def load_and_stack_datasets(base_path="Data/Box_data_NL2/"):
    
    # Build file paths using the base_path variable
    f1_VA    = sorted(glob(f'{base_path}box.nl2*2010*_VAO.nc'))[0:-1:2]
    f1_VNL   = sorted(glob(f'{base_path}box.nl2*2010*_VNL.nc'))[0:-1:2]
    f1_CG    = sorted(glob(f'{base_path}box.nl2*2010*_CG.nc'))[0:-1:2]
    f1_field = sorted(glob(f'{base_path}box.nl2*2010*Z.nc'))[0:-1:2]

    # Open and load datasets
    ds_f   = xr.open_mfdataset(f1_field).load()
    ds_VA  = xr.open_mfdataset(f1_VA).load()
    ds_VNL = xr.open_mfdataset(f1_VNL).load()
    ds_CG  = xr.open_mfdataset(f1_CG).load()

    # Stack datasets along a new dimension 'time_station'
    ds_f_stack   = ds_f.stack(time_station=['time', 'station'])
    ds_VA_stack  = ds_VA.stack(time_station=['time', 'station'])
    ds_VNL_stack = ds_VNL.stack(time_station=['time', 'station'])
    ds_CG_stack  = ds_CG.stack(time_station=['time', 'station'])
    
    # Load file_nk from the provided base_path
    file_nk = np.fromfile(f'{base_path}sig_nk.dat', sep='\t', dtype=float)
    
    return ds_f_stack, ds_VA_stack, ds_VNL_stack, ds_CG_stack, file_nk

# =============================================================================
# Filter Data Based on VNL Maximum
# =============================================================================
def filter_vnl_maximum(ds_VNL_stack, ds_VA_stack, ds_CG_stack, ds_f_stack, epsilon=-7):
    """
    Computes the maximum VNL value per time station, calculates an exponent factor,
    assigns it to the VNL dataset, and selects only time stations where the factor equals 1.
    
    Parameters:
        epsilon (float): The threshold to set exponent factors low enough to 0.
    """
    # Compute maximum VNL value along the 'nspec' dimension
    max_V = ds_VNL_stack.VNL.max(dim=['nspec'])
    max_V_exp = np.floor(np.log10(np.abs(max_V.values)))
    max_V_exp[max_V_exp <= epsilon] = 0
    max_V_exp[max_V_exp < 0] = 1

    # Assign the factor to the dataset
    ds_VNL_stack = ds_VNL_stack.assign(factor=(['time_station'], max_V_exp))
    
    # Select time stations with factor == 1
    ts = ds_VNL_stack.time_station.where(ds_VNL_stack.factor == 1, drop=True)
    ds_VA_stack  = ds_VA_stack.sel(time_station=ts)
    ds_VNL_stack = ds_VNL_stack.sel(time_station=ts)
    ds_CG_stack  = ds_CG_stack.sel(time_station=ts)
    ds_f_stack   = ds_f_stack.sel(time_station=ts)
    
    return ds_f_stack, ds_VA_stack, ds_VNL_stack, ds_CG_stack


# =============================================================================
# Convert and Reshape Data
# =============================================================================
def conv_fd_data(ds_VA_stack, ds_VNL_stack, file_nk):
  
    # Determine the number of samples (columns in the original data)
    num_samples = ds_VA_stack.VA.shape[1]

    # --- Reshape VA Data ---
    VNL2D_VA = np.empty([num_samples, 50, 36], dtype=float)
    for ik in range(1, 51):  # Frequency bins: 1 to 50
        for ith in range(1, 37):  # Directional bins: 1 to 36
            ISP = ith + (ik - 1) * 36
            VNL2D_VA[:, ik - 1, ith - 1] = ds_VA_stack.VA.values[ISP - 1, :]
    ds_VA_stack2 = ds_VA_stack.copy()
    new_shape_VA = (ds_VA_stack.VA.shape[1], 50 * 36)
    ds_VA_stack2 = ds_VA_stack2.assign(
        VA=(['nspec', 'time_station'], np.reshape(VNL2D_VA, new_shape_VA, order='C').T)
    )

    # --- Reshape VNL Data ---
    VNL2D_VNL = np.empty([ds_VNL_stack.VNL.shape[1], 50, 36], dtype=float)
    for ik in range(1, 51):
        for ith in range(1, 37):
            ISP = ith + (ik - 1) * 36
            VNL2D_VNL[:, ik - 1, ith - 1] = ds_VNL_stack.VNL.values[ISP - 1, :]
    ds_VNL_stack2 = ds_VNL_stack.copy()
    new_shape_VNL = (ds_VNL_stack.VNL.shape[1], 50 * 36)
    ds_VNL_stack2 = ds_VNL_stack2.assign(
        VNL=(['nspec', 'time_station'], np.reshape(VNL2D_VNL, new_shape_VNL, order='C').T)
    )
    
    return ds_VA_stack2, ds_VNL_stack2


# =============================================================================
# Normalization Helper Functions
# =============================================================================
def compute_stats(ds, var, dims):
    mean = ds[var].mean(dim=dims)
    std = ds[var].std(dim=dims)
    return mean, std

def normalize(ds, var, mean, std):
    return (ds[var] - mean) / std


# =============================================================================
# Combine Input Data Using PyTorch (Excluding nk_all)
# =============================================================================
def combine_input_data(ds_VA_norm, ds_f_dpt_norm, ds_CG_norm):
    # Convert xarray DataArrays to PyTorch tensors
    VA    = torch.tensor(ds_VA_norm.VA.values)
    f_dpt = torch.tensor(ds_f_dpt_norm.values)
    CG    = torch.tensor(ds_CG_norm.rename_dims({'nk': 'nspec'}).CG.values)
    
    # Concatenate along the channel dimension (first axis)
    combined = torch.cat([VA, f_dpt, CG], dim=0)
    return combined


# =============================================================================
# Main: Split, Normalize, Combine, and Save Processed Data
# =============================================================================
def main():
    # --- Load and Stack Datasets ---
    ds_f_stack, ds_VA_stack, ds_VNL_stack, ds_CG_stack, file_nk = load_and_stack_datasets()

    # Create a permutation of indices for splitting
    N = ds_VA_stack.time_station.size
    indices = np.random.permutation(np.arange(N))
    
    # --- Filter Data Based on VNL Maximum ---
    ds_f_stack, ds_VA_stack, ds_VNL_stack, ds_CG_stack = filter_vnl_maximum(
        ds_VNL_stack, ds_VA_stack, ds_CG_stack, ds_f_stack, epsilon=-7
    )
    
    # --- Convert to frequency-direction ---
    ds_VA_stack, ds_VNL_stack = conv_fd_data(ds_VA_stack, ds_VNL_stack, file_nk)
    
    # --- Split Dataset into Training and Testing Sets ---
    split_idx = int(0.7 * N)
    ds_VA_train  = ds_VA_stack.isel(time_station=indices[:split_idx])
    ds_VNL_train = ds_VNL_stack.isel(time_station=indices[:split_idx])
    ds_CG_train  = ds_CG_stack.isel(time_station=indices[:split_idx])
    ds_f_train   = ds_f_stack.isel(time_station=indices[:split_idx])
    
    ds_VA_test   = ds_VA_stack.isel(time_station=indices[split_idx:])
    ds_VNL_test  = ds_VNL_stack.isel(time_station=indices[split_idx:])
    ds_CG_test   = ds_CG_stack.isel(time_station=indices[split_idx:])
    ds_f_test    = ds_f_stack.isel(time_station=indices[split_idx:])
    
    # --- Compute normalization parameters of training data ---
    VA_mean, VA_std       = compute_stats(ds_VA_train, 'VA', ['time_station'])
    VNL_mean, VNL_std     = compute_stats(ds_VNL_train, 'VNL', ['time_station'])
    CG_mean, CG_std       = compute_stats(ds_CG_train, 'CG', ['nk', 'time_station'])
    wnd_mean, wnd_std     = compute_stats(ds_f_train, 'wnd', ['time_station'])
    dpt_mean, dpt_std     = compute_stats(ds_f_train, 'dpt', ['time_station'])
    
    # Build norm_data
    norm_data = xr.Dataset({
        'VA_mean': VA_mean,   'VA_std': VA_std,
        'VNL_mean': VNL_mean, 'VNL_std': VNL_std,
        'CG_mean': CG_mean,   'CG_std': CG_std,
        'wnd_mean': wnd_mean, 'wnd_std': wnd_std,
        'dpt_mean': dpt_mean, 'dpt_std': dpt_std,
    })
    
    # --- Normalize the Datasets ---
    ds_VA_norm       = normalize(ds_VA_train, 'VA', VA_mean, VA_std)
    ds_VNL_norm      = normalize(ds_VNL_train, 'VNL', VNL_mean, VNL_std)
    ds_CG_norm       = normalize(ds_CG_train, 'CG', CG_mean, CG_std)
    ds_f_norm        = normalize(ds_f_train, 'wnd', wnd_mean, wnd_std)
    ds_f_dpt_norm    = normalize(ds_f_train, 'dpt', dpt_mean, dpt_std)
    
    ds_VA_norm_test    = normalize(ds_VA_test, 'VA', VA_mean, VA_std)
    ds_VNL_norm_test   = normalize(ds_VNL_test, 'VNL', VNL_mean, VNL_std)
    ds_CG_norm_test    = normalize(ds_CG_test, 'CG', CG_mean, CG_std)
    ds_f_norm_test     = normalize(ds_f_test, 'wnd', wnd_mean, wnd_std)
    ds_f_dpt_test      = normalize(ds_f_test, 'dpt', dpt_mean, dpt_std)
    
    # --- Combine input datasets for training and testing ---
    x_tensor = combine_input_data(ds_VA_norm, ds_f_dpt_norm, ds_CG_norm)
    x_valid_tensor = combine_input_data(ds_VA_norm_test, ds_f_dpt_test, ds_CG_norm_test)
    
    # output dataset
    y_tensor = torch.FloatTensor(ds_VNL_norm.VNL.transpose().values)
    y_valid_tensor = torch.FloatTensor(ds_VNL_norm_test.VNL.transpose().values)
    
    # --- Save Processed Data ---
    torch.save(x_tensor, 'Data/xtrain_nl2.pt')
    torch.save(y_tensor, 'Data/ytrain_nl2.pt')
    torch.save(x_valid_tensor, 'Data/x_valid_nl2.pt')
    torch.save(y_valid_tensor, 'Data/y_valid_nl2.pt')
    norm_data.to_netcdf('Data/norm_data_nl2.nc')
    
    print("Data generation and saving complete.")

if __name__ == '__main__':
    main()

