# document-based-question-answering

# run

```bash
python3 -m venv .venv
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
pip install -e .
```

# Other requirements

## Llama

## vicuna

Use this conversion [script](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py)

```
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```

Run on huggingface weights after conversion

```
python3 -m fastchat.model.apply_delta \
    --base-model-path /path/to/llama-7b \
    --target-model-path /path/to/output/vicuna-7b \
    --delta-path lmsys/vicuna-7b-delta-v1.1
```
