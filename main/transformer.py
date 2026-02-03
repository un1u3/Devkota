import torch 
import torch.nn as nn 
from src.core.multi_head_attention import MultiHeadAttention, create_casual_mask
from src.core.feedforward import FeedForward



class TransformerBlock(nn.Module):
    # A single transformer decoder block 
    # Arch :  LayerNorm --> self-attn -->residual conncetion 
    #         Layer-> feedforward->residual connecction 
