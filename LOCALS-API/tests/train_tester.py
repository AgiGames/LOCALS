from locals import LOCALS, LOCALSDataset

model = LOCALS()
dataset = LOCALSDataset('dataset/images', 'dataset/labels')
train_loader, test_loader, val_loader = dataset.get_dataloaders(train_split=0.9, test_split=0.1)
model.fit(train_loader, num_epochs=2, val_loader=val_loader)

print("\033[92mSUCCESS\033[0m")