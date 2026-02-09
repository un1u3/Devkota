import torch 
from torch.utils.data import DataSet, DataLoader

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sequences = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                tokens = tokenizer.encode(line, add_bos=True, add_eos=True)
                
                if len(tokens) > max_len:
                    tokens = tokens[:max_len]
                
                if len(tokens) > 10:  # Skip very short
                    self.sequences.append(tokens)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Input and target (shifted by 1)
        input_ids = seq[:-1]
        labels = seq[1:]
        
        # Pad
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_id] * pad_len
            labels = labels + [self.tokenizer.pad_id] * pad_len
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
def get_dataloaders(train_path, val_path, tokenizer, batch_size=8, max_len=512):
    train_dataset = TextDataset(train_path, tokenizer, max_len)
    val_dataset = TextDataset(val_path, tokenizer, max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader