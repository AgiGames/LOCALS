import torch
import numpy as np
from torch.utils.flop_counter import FlopCounterMode

import os
import json
import uuid

from .locals import LOCALS
from .dataset import LOCALSDataset

def seeder(seed: int):
    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def save_run_metrics(run_details: dict):
    os.makedirs('runs', exist_ok=True)
    
    run_id = str(uuid.uuid4())
    
    with open(f'runs/run_metrics_{run_id}.json', 'w') as run_metrics_file:
        json.dump(run_details, run_metrics_file)

def run(*, seed: int,
        train_split: float,
        test_split: float,
        images_dir: str,
        labels_dir: str,
        architecture='n',
        num_epochs=50,
        epoch_to_lr={}):
    
    seeder(seed)
    
    dataset = LOCALSDataset(images_dir, labels_dir)
    train_loader, test_loader, val_loader = dataset.get_dataloaders(train_split, test_split)
    
    model = LOCALS(architecture)
    model.fit(train_loader, num_epochs=num_epochs, val_loader=val_loader, epoch_to_lr=epoch_to_lr)
    
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        for images, labels in test_loader:
            outputs = model(images.to(model.device))
            break
    total_flops = flop_counter.get_total_flops()
    
    metrics= model.evaluate(test_loader)
    run_details = {'seed': seed,
                   'train_split': train_split,
                   'test_split': test_split,
                   'recall': metrics['recall'],
                   'precision': metrics['precision'],
                   'f1_score': metrics['f1_score'],
                   'mAP': metrics['mAP'],
                   'mCS': metrics['mCS'],
                   'num_params': model.get_num_params(),
                   'total_flops': total_flops}
    
    save_run_metrics(run_details)
    return run_details