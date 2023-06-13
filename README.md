# document-based-question-answering

# System specs
- python version : `3.10.11`
- system : `Ubuntu 20.04.6 LTS x86_64`
- GPU : `1x NVIDIA GeForce GTX 1080 Ti and 2x NVIDIA GeForce RTX 3060`
- RAM : `126GB`

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

## Llama 7B
we are using [OpenLlama](https://github.com/openlm-research/open_llama) and open reproduction of Llama, weights for the 7B model are available on [here](https://huggingface.co/openlm-research/open_llama_7b).

```bash
git lfs install
git clone https://huggingface.co/openlm-research/open_llama_7b models/openllama/7B
```

## Vicuna 7B

- Use this conversion [script](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py) on the consolidated weights to convert them to huggingface format.

```bash
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
```

- Apply the delta weights of vicuna on llama weights (huggingface format, i.e after conversion).

```bash
python3 -m fastchat.model.apply_delta \
    --base-model-path /path/to/llama-7b \
    --target-model-path /path/to/output/vicuna-7b \
    --delta-path lmsys/vicuna-7b-delta-v1.1
```

## Falcon 7B


## datasets

- [Google natural questions](https://ai.google.com/research/NaturalQuestions/download) (we are using the dev set)

```bash
gzip -d dataset/v1.0-simplified_nq-dev-all.jsonl.gz 
```
sample a smaller dataset

```bash
python scripts/sample_qa.py 
```

## Test
- you can test if the models are working by running :
```bash
python scripts/run_llama.py
python scripts/run_vicuna.py
python scripts/fly_falcon.py
```


# Notes

- temperature of vicuna is set to 0
- temperature of falcon is set to 3e-4
- temperature of llama is set to 0
