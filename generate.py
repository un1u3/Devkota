import torch
from main.devkota import Devkota
from src.core.config import ModelConfig
from src.core.train_spm import NepaliTokenizer
from src.core.utils import load_checkpoint
import os

def generate_text(prompt, model, tokenizer, max_tokens=100, temperature=0.8):
    # wrapper for generation 
    
    # encode prompt 
    input_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    
    # device thing 
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    # generate 
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=50,
        top_p=0.9
    )
    
    # decode back to text
    generated_text = tokenizer.decode(output_ids[0].tolist())
    return generated_text

def main():
    # config setup
    model_cfg = ModelConfig()
    
    # model init 
    model = Devkota(
        vocab_size=model_cfg.vocab_size,
        d_model=model_cfg.d_model,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        d_ff=model_cfg.d_ff,
        max_seq_len=model_cfg.max_seq_len,
        dropout=model_cfg.dropout,
        pad_idx=model_cfg.pad_idx
    )
    
    # loading checkpoint 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = "checkpoints/best_model.pt"
    
    if os.path.exists(checkpoint_path):
        print(f"loading from {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, model)
        print("model loaded")
    else:
        print("checkpoint xina, using untrained model")

        
    model = model.to(device)
    model.eval()
    
    # tokenizer 
    tokenizer_path = "tokenizer/devkota_tokenizer.model"
    if not os.path.exists(tokenizer_path):
        print("tokenizer missing")
        return

    tokenizer = NepaliTokenizer(tokenizer_path)
    
    print("Devkota Generator")
    print("type 'quit' to exit")
    
    while True:
        try:
            prompt = input("Enter prompt: ")
            if prompt.strip() == 'quit':
                break
            
            print("generating...")
            generated = generate_text(
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                max_tokens=100
            )
            
            print(f"\nResult: {generated}\n")
            print("-" * 50)
        except KeyboardInterrupt:
            # quit on ctrl c 
            break
        except Exception as e:
            print(f"error: {e}")

if __name__ == "__main__":
    main()
