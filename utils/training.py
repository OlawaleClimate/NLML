import torch

def train(model, criterion, train_loader, validation_loader, optimizer, epochs, fname_prefix, device, save_epochs=None):
    if save_epochs is None:
        save_epochs = [100, 300, 500, 800, 1000, 1500, 2000, 2500, 3000]
        
    training_stats = {'train_loss': [], 'valid_loss': []}
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            # Adjust input slice if needed (using fixed input_size from config)
            x = x[:, :1851].to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        training_stats['train_loss'].append(train_loss / len(train_loader))
        
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for x, y in validation_loader:
                x = x[:, :1851].to(device)
                y = y.to(device)
                output = model(x)
                loss = criterion(output, y)
                valid_loss += loss.item()
        training_stats['valid_loss'].append(valid_loss / len(validation_loader))
        
        if epoch in save_epochs:
            torch.save(model, f"{fname_prefix}{epoch}ep.pt")
            print(f"Model saved at epoch {epoch}")
    return training_stats

