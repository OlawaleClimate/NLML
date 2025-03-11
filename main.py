# main.py
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import xarray as xr
from torchsummary import summary

from data.data_loader import MyDataset, load_data
from models.model import CustomNN
from utils.training import train

def main():
    # === Configuration ===
    config = {
        'input_size': 1851,
        'output_size': 1800,
        'hidden_layers': 4,
        'neurons_per_layer': [3072],  # Base neuron count; will be repeated hidden_layers times.
        'dropout_prob': 0.0,
        'use_batch_norm': False,
        'activation': 'leaky_relu',
        'learning_rate': 1e-4,
        'epochs': 301,
        'batch_size': 1024,
        'model_save_prefix': './'
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Multiply the base neuron count by the number of layers
    neurons_list = config['neurons_per_layer'] * config['hidden_layers']
    
    # === Data Loading ===
    x, y, x_valid, y_valid, norm_data = load_data()
    train_dataset = MyDataset(x, y)
    valid_dataset = MyDataset(x_valid, y_valid)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True)
    
    # === Model Initialization ===
    model = CustomNN(
        input_size=config['input_size'],
        output_size=config['output_size'],
        hidden_layers=config['hidden_layers'],
        neurons_per_layer=neurons_list,
        dropout_prob=config['dropout_prob'],
        use_batch_norm=config['use_batch_norm'],
        activation=config['activation']
    ).to(device)
    
    # Display model summary
    summary(model, (config['batch_size'], config['input_size']))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], amsgrad=True)
    
    # === Custom Loss Function ===

    # Create normalization tensor (if needed later in training)
    fac = 1e6
    a1 = (fac * norm_data['VNL_std'] + 0).transpose().values
    norm_tensor = torch.FloatTensor(a1).to(device)
    
    def criterion(output, target):
         return torch.mean(torch.abs(norm_tensor*(target - output)))
    
    # === Training ===
    start_time = time.time()
    training_results = train(
        model, criterion, train_dataloader, valid_dataloader,
        optimizer, config['epochs'], config['model_save_prefix'], device
    )
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    
    # Save training logs
    model_save_name = f"{config['model_save_prefix']}L{config['hidden_layers']}_{neurons_list[0]}_{config['epochs']}epoch"

    np.savetxt(config['model_save_name'] + 'tr.txt', np.array(training_results['train_loss']))
    np.savetxt(config['model_save_name'] + 'vd.txt', np.array(training_results['valid_loss']))

if __name__ == '__main__':
    main()

