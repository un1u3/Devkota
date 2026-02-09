# Training Configuration

class ModelConfig:
    vocab_size = 16000
    d_model = 512
    num_layers = 12
    num_heads = 8
    d_ff = 2048
    max_seq_len = 512
    dropout = 0.1
    pad_idx = 3


class PreTrainConfig:
    # Data
    train_data = "preprocessed_data/train.txt"
    val_data = "preprocessed_data/val.txt"
    
    # Training
    batch_size = 8
    accumulation_steps = 4
    epochs = 10
    
    # Optimizer
    lr = 3e-4
    weight_decay = 0.01
    warmup_steps = 2000
    max_grad_norm = 1.0
    
    # Checkpoint
    checkpoint_dir = "checkpoints"
    save_every = 1000
    eval_every = 500


class FineTuneConfig:
    # Data
    train_data = "preprocessed_data/devkota_train.txt"
    val_data = "preprocessed_data/devkota_val.txt"
    pretrained = "checkpoints/best_model.pt"
    
    # Training
    batch_size = 4
    accumulation_steps = 2
    epochs = 20
    
    # Optimizer
    lr = 1e-5
    weight_decay = 0.01
    warmup_steps = 100
    max_grad_norm = 1.0
    
    # Early stopping
    patience = 3
    
    # Checkpoint
    checkpoint_dir = "checkpoints/finetuned"
    save_every = 200
    eval_every = 100
    output = "checkpoints/devkota_poet.pt"