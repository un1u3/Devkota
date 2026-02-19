# Devkota 
Small Nepali decoder-only Transformer that you can pretrain, fine-tune on poetry, and generate poems from prompts.

## Quick Start
- Install deps: `pip install -r requirements.txt`
- Train tokenizer: `python3 train_tokenizer.py`
- Pretrain (news corpus): `python3 train.py`
- Fine-tune on poems (provide your own): put poetry lines in `preprocessed_data/devkota_train.txt` and `preprocessed_data/devkota_val.txt`, then `python3 finetune.py`
- Generate poems: `python3 generate.py` and enter a theme like `पहाड`
- Streamlit demo: `streamlit run app_streamlit.py`

## Project Map
- Model: `main/devkota.py`, `main/transformer.py`
- Training: `train.py` (pretrain), `finetune.py` (poetry), `src/core/trainer.py`
- Tokenizer: `src/core/train_spm.py`, `train_tokenizer.py`
- Data loaders: `src/core/dataset.py`
- Configs: `src/core/config.py`
- Generation CLI: `generate.py`

## Notes
- CUDA auto-uses if available; CPU works but is slow.
- Checkpoints: pretrain → `checkpoints/best_model.pt`; poetry → `checkpoints/finetuned/devkota_poet.pt` (used by `generate.py` if present).
- Outputs improve dramatically after fine-tuning on a real poetry corpus.

![alt text](image.png)

