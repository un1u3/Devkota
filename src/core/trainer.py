import torch 
import torch.nn as nn 
from torch.cuda.amp import autocase, GradScaler
import time 
from pathlib import Path 
from utils import LRScheduler, save_checkpoint, load_checkpoint, compute_preplx

class Trainer:
    def __init__(self, model, train_loader, val_loader config, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config 
        self.device = device

        # usign AdamW(standar) will try more if needed 
        self.optimizer = torch.optim.AdamW(model.parameters(),lr = config.lr, weight_decay = config.weight_decay)

        steps_per_epoch = len(train_loader) // config.accumulation_steps
        total_steps = steps_per_epoch * config.epochs

        # warmup + cosine decay scheduler driven by optimzer step count 
        self.scheduler = LRScheduler(
            self.optimizer,
            peak_lr=config.lr,
            warmup_steps= config.warmup_steps,
            total_steps=total_steps
        )

        # GrandScaler dynamically resccales loss to prevent underflwo 
        self.scaler = GradScaler()
        
        # global optimization step counter 
        self.step = 0
        self.epoch = 0

        # track best validation loss for model selection 
        self.best_val_loss = float('inf')

        # ensure checkpoint directory exists before training starts 
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    
        