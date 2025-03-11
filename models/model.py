import torch.nn as nn

class CustomNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, neurons_per_layer, 
                 dropout_prob=0.0, use_batch_norm=False, activation='relu'):
        super(CustomNN, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_size, neurons_per_layer[0]))
        self.layers.append(self.get_activation(activation))
        
        # Hidden layers
        for i in range(hidden_layers):
            in_features = neurons_per_layer[i]
            # Use the next element in the list if available; otherwise, reuse the same
            out_features = neurons_per_layer[i+1] if i < len(neurons_per_layer) - 1 else neurons_per_layer[i]
            self.layers.append(nn.Linear(in_features, out_features))
            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(out_features, track_running_stats=False))
            self.layers.append(self.get_activation(activation))
            if dropout_prob > 0:
                self.layers.append(nn.Dropout(dropout_prob))
        
        # Output layer
        self.layers.append(nn.Linear(neurons_per_layer[-1], output_size))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

