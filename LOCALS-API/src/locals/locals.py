import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode

from .trainer import train
from .architectures import LOCALSn, LOCALSs
from .figmaker import save_loss_fig, save_recall_precision_f1_score
from .evaluator import find_recall_precision_f1_score, find_mAP, find_mCS

class LOCALS:
    def __init__(self, architecture='n', file_path=None, device='cuda'):
        self.architecture = architecture
        self.device = device
        if architecture == 'n':
            self.model = LOCALSn()
            self.model.to(device)
        if file_path:
            self.model.load_state_dict(torch.load(file_path, weights_only=True))
            
    def __call__(self, images):
        return self.model(images)
            
    def fit(self, train_loader: DataLoader, *, num_epochs=50, val_loader=None, epoch_to_lr={}, do_save_loss_fig=True):
        train_loss_ot, val_loss_ot = train(train_loader, num_epochs, val_loader, self.model, self.device, epoch_to_lr)
        self.model.load_state_dict(torch.load("best.pth", weights_only=True))
        
        if do_save_loss_fig:
            save_loss_fig(train_loss_ot, val_loss_ot)
            
    def eval(self):
        self.model.eval()
    
    def train(self):
        self.model.train()
        
    def evaluate(self, data, do_save_fig=True):
        recall, precision, f1_score = find_recall_precision_f1_score(self, data)
        mAP = find_mAP(self, data)
        mCS = find_mCS(self, data)
            
        if do_save_fig:
            save_recall_precision_f1_score(recall, precision, f1_score)
        
        metrics_dataframe = pd.Series({'recall': recall, 'precision': precision, 'f1_score': f1_score, 'mAP': mAP, 'mCS': mCS})
        print(metrics_dataframe)
        return metrics_dataframe
    
    def get_num_params(self):
        num_params = sum(
            p.numel()
            for p in self.model.parameters()
        )
        return num_params 
    
    def get_num_flops(self, display=False):
        self.eval()
        
        flop_counter = FlopCounterMode(display=display)
        dummy_tens = torch.rand(1, 3, 448, 448, device=self.device)
        
        with flop_counter:
            self.model(dummy_tens)
            
        return flop_counter.get_total_flops()