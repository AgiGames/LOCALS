import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader

from .losses import locals_loss

def train(train_loader: DataLoader,
          num_epochs: int,
          val_loader: DataLoader,
          model: torch.nn.Module,
          device='cuda',
          epoch_to_lr = {}):
    
    model.to(device)
    torch.save(model.state_dict(), "best.pth")
    
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = locals_loss()
    validate = val_loader is not None
    
    train_loss_ot = []
    val_loss_ot = []
    avg_train_loss = 0
    avg_val_loss = 0
    min_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        if epoch in epoch_to_lr:
            new_lr = epoch_to_lr[epoch]
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr
            if validate:
                model.load_state_dict(torch.load("best.pth", weights_only=True))
            
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")

        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        train_loss_ot.append(avg_train_loss)
        print(f'Avg Training Loss = {avg_train_loss}')
            
        if validate:
            model.eval()
            with torch.no_grad():
                total_val_loss = 0

                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()
                    
            avg_val_loss = total_val_loss / len(val_loader)
            val_loss_ot.append(avg_val_loss)
            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                torch.save(model.state_dict(), "best.pth")
            
            print(f'Val Loss = {avg_val_loss}')
    
    if not validate:
        torch.save(model.state_dict(), "best.pth")
        
    return train_loss_ot, val_loss_ot