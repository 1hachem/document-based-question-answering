#! /bin/bash
source .venv/bin/activate
python -m fastchat.serve.cli --model-path models/vicuna/7B --num-gpus 3