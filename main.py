import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchsummary import summary

# Import data generator and data loader functions
from data.data_generator import main as generate_data
from data.data_loader import load_data, MyDataset
from models.custom_nn import CustomNN
from utils.training import train

def main():
    # Configuration dictionary: change values as needed.
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
    
    # Updated list of data files
    data_files = [
        'Data/xtrain_nl2.pt',
        'Data/ytrain_nl2.pt',
        'Data/x_valid_nl2.pt',
        'Data/y_valid_nl2.pt',
        'Data/norm_data_nl2.nc'
    ]
    
    # Check if data files exist; if not, generate them.
    if not all(os.path.exists(f) for f in data_files):
        print("Generating data...")
        generate_data()
        print("Data generation complete.")
    else:
        print("Data already generated. Skipping generation.")

    # --- Load Data ---
    x, y, x_valid, y_valid, norm_data = load_data()

    # --- Create Dataset and Dataloaders ---
    train_dataset = MyDataset(x, y)
    valid_dataset = MyDataset(x_valid, y_valid)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True)

    # --- Model Configuration ---
    neurons_list = config['neurons_per_layer'] * config['hidden_layers']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Build Model ---
    model = CustomNN(
        config['input_size'],
        config['output_size'],
        config['hidden_layers'],
        neurons_list,
        config['dropout_prob'],
        config['use_batch_norm'],
        config['activation']
    ).to(device)
    
    summary(model, (config['batch_size'], config['input_size']))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], amsgrad=True)
    
    # === Custom Loss Function ===
    # Create normalization tensor (if needed later in training)
    a1 = (norm_data['VNL_std'] + 0).transpose().values
    norm_tensor = torch.FloatTensor(a1).to(device)
    
    def criterion(output, target):
         return torch.mean(torch.abs(norm_tensor * (target - output)))

    # Create dynamic model save name using config parameters:
    model_save_name = f"{config['model_save_prefix']}L{config['hidden_layers']}_{neurons_list[0]}_"
    
    # --- Train Model ---
    start_time = time.time()
    training_results = train(
        model, criterion, train_dataloader, valid_dataloader,
        optimizer, config['epochs'], model_save_name, device
    )
    elapsed = time.time() - start_time
    print("Training completed in {:.2f} seconds".format(elapsed))
    
    
    
    # Save training logs
    model_save_name = f"{config['model_save_prefix']}L{config['hidden_layers']}_{neurons_list[0]}_{config['epochs']}epoch"
    np.savetxt(model_save_name + '_tr.txt', np.array(training_results['train_loss']))
    np.savetxt(model_save_name + '_vd.txt', np.array(training_results['valid_loss']))

if __name__ == '__main__':
    main()

