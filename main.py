# nohup bash -c "python main.py --dataset datasets/content_moderation_lm3.jsonl --model_id meta-llama/Llama-Guard-3-8B --device cuda:0" > out.log 2>&1 &

## Imports

import argparse

import pandas as pd
import torch
import yaml
from src.config import Config
from src.metrics import compute_metrics
from src.runner import Runner
from transformers import AutoModelForCausalLM, AutoTokenizer

## Parse args
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    choices=[
        "datasets/content_moderation_lm2.jsonl",
        "datasets/content_moderation_lm3.jsonl",
        "beavertails",
    ],
    default="datasets/content_moderation_lm3.jsonl",
    help="Dataset to use for evaluation",
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=0,
    help="Number of samples to use for evaluation",
)
parser.add_argument(
    "--model_id",
    type=str,
    choices=[
        "meta-llama/Llama-Guard-3-8B",
        "meta-llama/Meta-Llama-Guard-2-8B",
        "meta-llama/Llama-3.1-8B-Instruct",
    ],
    default="meta-llama/Llama-Guard-3-8B",
    help="Model identifier",
)
parser.add_argument(
    "--device", type=str, default="cuda:0", help="Device to run the model on"
)
args = parser.parse_args()

model_id = args.model_id
device = args.device
dtype = torch.bfloat16

## Load model and tokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=dtype, device_map=device
)
config = Config(args, model, tokenizer)

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)
strategy_params = params["strategies"]
model_overrides = params.get("models", {})

runner = Runner(model, tokenizer, config)
metrics_list = []
for strategy, params in strategy_params.items():
    strat_params = {
        **params,
        **model_overrides.get(model_id, {}),
    }
    processed_dataset = runner.run(
        strategy=strategy,
        **strat_params,
    )
    metrics = compute_metrics(
        processed_dataset, config.labels, threshold=strat_params.get("threshold", 0.2)
    )
    metrics_list.append(
        {
            "Strategy": strategy,
            **metrics,
        }
    )

metrics_df = pd.DataFrame(metrics_list)
metrics_df.set_index(["Strategy"], inplace=True)
print(metrics_df)
