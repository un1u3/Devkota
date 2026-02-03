import torch 
import torch.nn as nn 
import torch.nn fucntional as F 
from src.core.positionalencoder import PositionalEncoding
from src.core.multi_head_attention import MultiHeadAttention, create_casual_mask, create_padding_mask
from transformer import TransformerBlock


class Devkota(nn.Module):
    # architecture
    # 1.TOken embedding 
    # 2.positional encoding 
    # 3. 12 transformer blocks 
    # 4.Finallayer norm 
    # lang model head (projectt to vocab)


    # ig it needs some value
    def __init__(self, vocab_size, d_model,num_layers, num_heads,d_ff, max_seq_len, dropout, pad_idx):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx

        # positional encodingn 
        # adds postion info to embedding 



    
        