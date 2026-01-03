# imports
import torch
import torch.nn as nn


# component 1 : TOKENIZER 
class Tokenizer:
    """
    Character -Level tokenizer for Nepali text 
    what it does 
    - Builds a vocabulart for all uqnieu characters in the text
    - Convert text to numbers(encoding) and vice versa 
    """
    def __init__(self,text):

        # process : extract->sort->create bidirectional mapping

        # 1. get all unique characterrs from the text
        self.chars = sorted(list(set(text)))
        # 2. count how many unqiue characters from the text 
        self.vocab_size = len(self.chars)
        # encoding 
        self.stoi = {} # start withh an empty dictionary 
        for i,ch in enumerate(self.chars): # go through each character with index 
            self.stoi[ch] = i # assign the index t the character 
        
        # decoding 
        self.itos = {}
        for i,ch in enumerate(self.chars):
            self.itos[i] = ch 
        
    def encode(self,text):
        indices = []
        for c in text:
            indices.append(self.stoi[c])
        return indices

    def decode(self,indices):
        chars = []
        for i in indices:
            chars.append(self.itos[i])
        return ''.join(chars)


# component 2: TOKEN EMBEDINGS 
class TokenEmbedding(nn.Module):
    # Converts token indices to dense vector representaions.
    # What it does 
        # - each character index -> D-dimenssional vector 
        # - vectors are learned during training 
        # - similar characters get similar vectors 
    
    def __init__(self, vocab_size, embedding_dim):
        # args
        # vocab_size = num of unique tokens(chars)
        # embedding_dim : size of embedding vectors 

        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim 
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # creates the embedding layer 
        # this is essentially a lookup table (matrix)

    def forward(self, token_indices):
        return self.embedding(token_indices)


# Fix PositionalEmbedding.forward (seq_len was not defined) and replace the model's pos_emb
class PositionalEmbedding(nn.Module):
    
    def __init__(self, max_seq_len, embedding_dim):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(max_seq_len, embedding_dim)

    def forward(self, token_embeddings):
        # token_embeddings: (batch_size, seq_len, embedding_dim) or (seq_len, embedding_dim)
        if token_embeddings.dim() == 3:
            seq_len = token_embeddings.size(1)
        elif token_embeddings.dim() == 2:
            seq_len = token_embeddings.size(0)
            token_embeddings = token_embeddings.unsqueeze(0)  # add batch dim for consistent addition
        else:
            raise ValueError(f"Unexpected token_embeddings shape: {token_embeddings.shape}")

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence Length {seq_len} exceeds maximum {self.max_seq_len}"
            )

        positions = torch.arange(seq_len, device=token_embeddings.device)
        pos_emb = self.embedding(positions)  # (seq_len, embedding_dim)

        # broadcasting will add pos_emb (seq_len, emb_dim) to token_embeddings (batch, seq_len, emb_dim)
        return token_embeddings + pos_emb


# component  4: COMBINED EMBEDDING LAYER
class NepaliEmbedding(nn.Module):
    # token + position 
    def __init__(self, vocab_size, embedding_dim, max_seq_len):
        # Initialize combined embedding layer.
        super().__init__()

        # store configuration 
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim 
        self.max_seq_len = max_seq_len 

        # create token embeding layer 
        self.token_emb = TokenEmbedding(vocab_size, embedding_dim)

        # create positiona embedding layer 
        self.pos_emb  = PositionalEmbedding(max_seq_len, embedding_dim)

        # calcualate total paramteres 
        total_parms = (vocab_size * embedding_dim) + (max_seq_len * embedding_dim)
    
    def forward(self, token_indices):
        # convert tokenn indices to embeddings with postion embedding 

        # 1. get token embedding 
        tok_emb = self.token_emb(token_indices)

        # 2.Add positioal embedding
        embeddings = self.pos_emb(tok_emb)
        return embeddings
