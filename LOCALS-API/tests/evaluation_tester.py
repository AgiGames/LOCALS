from locals import LOCALS, LOCALSDataset, visualise_dataset, visualise_predictions

model = LOCALS('n', 'evaluation_model.pth')
dataset = LOCALSDataset('dataset/images', 'dataset/labels')
train_loader, test_loader, val_loader = dataset.get_dataloaders(train_split=0.9, test_split=0.1)

# visualise_dataset(train_loader, 8)
# visualise_predictions(model, test_loader, 8)

metrics = model.evaluate(test_loader)
print(f"Recall: {metrics['recall']}")
print(f"Precision: {metrics['precision']}")
print(f"F1 Score: {metrics['f1_score']}")
print(f"mAP: {metrics['mAP']}")
print(f"mCS: {metrics['mCS']}")

print("\033[92mSUCCESS\033[0m")