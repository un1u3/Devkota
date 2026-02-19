# Devkota Project Notes 
<i>Summarised by AI </i>


## What it is
- A small decoder-only Transformer language model (about 10M params) implemented from scratch in PyTorch, optimized for low-VRAM (8GB) training.
- Uses a SentencePiece BPE tokenizer (default vocab 16k).
- Training pipeline: pretrain on generic Nepali text, then fine-tune on poetry (POeLM idea), then generate poems from prompts.

## Key files
- Model: `main/devkota.py`, `main/transformer.py`
- Attention/FFN/PE: `src/core/multi_head_attention.py`, `src/core/feedforward.py`, `src/core/positionalencoder.py`
- Data pipeline: `src/core/dataset.py`
- Training loop: `src/core/trainer.py` (Trainer + FineTuner, AMP, grad accumulation, cosine LR, checkpoints)
- Tokenizer: `src/core/train_spm.py`, driver `train_tokenizer.py`
- Pretrain driver: `train.py`
- Fine-tune driver: `finetune.py`
- Generation CLI: `generate.py`
- Configs: `src/core/config.py` (ModelConfig, PreTrainConfig, FineTuneConfig)

## Data expected
- Pretrain: `preprocessed_data/train_small.txt` (100k lines) and `preprocessed_data/val_small.txt` (5k lines), news-style Nepali text.
- Poetry fine-tune: `preprocessed_data/devkota_train.txt` and `preprocessed_data/devkota_val.txt` (one poem/stanza per line). Provide your own poetry corpus.

## Training steps (from repo root)
- Install deps: `pip install -r requirements.txt`
- Train tokenizer: `python3 train_tokenizer.py` → creates `tokenizer/devkota_tokenizer.model`
- Pretrain LM: `python3 train.py` → saves checkpoints in `checkpoints/` (best at `checkpoints/best_model.pt`)
- Fine-tune on poetry: ensure poetry files exist, then `python3 finetune.py` → saves `checkpoints/finetuned/devkota_poet.pt`

## Generation
- Run: `python3 generate.py`
- It loads finetuned checkpoint if present, otherwise best pretrain.
- Use short prompts, e.g.:
  - `विषय: पहाड\nकविता (४ पङ्क्ति):`
- Sampling defaults (tuned to reduce repetition): `temperature=0.7`, `top_k=40`, `top_p=0.85`, `repetition_penalty=1.1`, `max_new_tokens=80`.
- Adjust in `generate.py` if output loops: lower temperature, raise repetition_penalty (e.g., 1.2), or shorten `max_new_tokens`.

## Current status / notes
- Works on CPU but slow; CUDA auto-enabled if available. GradScaler/autocast disabled on CPU.
- Base pretrain data is small → outputs are generic; poetry quality depends on your fine-tune corpus.
- Added repetition penalty and prompt-only decoding to reduce echoes.
- `run_laptop.sh` can create small pretrain splits if `data/preprocessed/input.txt` exists (not included).

## Quick file map for modifications
- Change hyperparams: `src/core/config.py`
- Change decoding settings: `generate.py` (function `generate_text`)
- Swap checkpoints: edit `checkpoint_path` logic in `generate.py` if needed.

## Suggested next steps
- Collect/clean a larger Nepali poetry set (>=8k train, 4k val lines already tested).
- Run fine-tune, then evaluate outputs; adjust decoding settings.
- For research quality: add control tokens for theme/meter, run human eval, and compare tokenizers (BPE vs unigram vs char).
