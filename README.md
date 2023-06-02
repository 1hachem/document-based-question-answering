# document-based-question-answering

# System specs
- python version : `3.10.11`
- system : `Ubuntu 20.04.6 LTS x86_64`
- GPU : `NVIDIA GeForce GTX 1080 Ti`

# Build
- create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```
- install the requirements
```bash
pip install -r requirements.txt
pip install -e .
```

# Other requirements

## Llama
- install the weights into `models/llama/7B` using this [script](https://github.com/Elyah2035/llama-dl/blob/main/llama.sh)

- follow the instructions at [llama.cpp](https://github.com/ggerganov/llama.cpp) to quantize the model (put the quantized model at `models/llama/7B/ggml-model-q4_0.bin`)

## Vicuna

- Use this conversion [script](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py) on the consolidated weights to convert them to huggingface format.

```bash
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```

- Apply the deltas weights of vicuna on llama weights (huggingface format, i.e after conversion).

```bash
python3 -m fastchat.model.apply_delta \
    --base-model-path /path/to/llama-7b \
    --target-model-path /path/to/output/vicuna-7b \
    --delta-path lmsys/vicuna-7b-delta-v1.1
```

## Test
- you can test of the models are working by running :
```bash
python scripts/run_llama.py
python scripts/run_vicuna.py
```
