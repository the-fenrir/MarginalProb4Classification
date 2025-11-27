import pandas as pd
from datasets import load_dataset
from src.benchmark_logtoku import benchmark_logtoku
from src.methods import compute_probability_for_labels
from src.utils import (
    beavertails_llama_guard_2_category_mapping,
    beavertails_llama_guard_3_category_mapping,
    read_jsonl,
)
from tqdm import tqdm


class Runner:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        if config.dataset.endswith(".jsonl"):
            self.dataset = pd.DataFrame(read_jsonl(config.dataset))
            self.dataset["conversation"] = self.dataset.apply(
                lambda x: [
                    {
                        "role": "user",
                        "content": x["unsafe_query"],
                    }
                ],
                axis=1,
            )
        elif config.dataset == "beavertails":
            self.dataset = (
                load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
                .to_pandas()
                .drop_duplicates(subset=["prompt", "response"])
                .dropna()
            )
            if config.model_id == "meta-llama/Meta-Llama-Guard-2-8B":
                self.dataset["llama_guard_2_categories"] = self.dataset[
                    "category"
                ].apply(
                    lambda x: list(
                        set(
                            [
                                beavertails_llama_guard_2_category_mapping[k]
                                for k, v in x.items()
                                if v
                            ]
                        )
                    )
                )
            else:
                self.dataset["llama_guard_3_categories"] = self.dataset[
                    "category"
                ].apply(
                    lambda x: list(
                        set(
                            [
                                beavertails_llama_guard_3_category_mapping[k]
                                for k, v in x.items()
                                if v
                            ]
                        )
                    )
                )
            self.dataset["conversation"] = self.dataset.apply(
                lambda x: [
                    {
                        "role": "user",
                        "content": x["prompt"],
                    },
                    {
                        "role": "assistant",
                        "content": x["response"],
                    },
                ],
                axis=1,
            )
        self.dataset["ground_labels"] = (
            self.dataset["llama_guard_2_categories"]
            if config.model_id == "meta-llama/Meta-Llama-Guard-2-8B"
            else self.dataset["llama_guard_3_categories"]
        )
        self.dataset = self.dataset[
            [
                (
                    "llama_guard_2_categories"
                    if config.model_id == "meta-llama/Meta-Llama-Guard-2-8B"
                    else "llama_guard_3_categories"
                ),
                "ground_labels",
                "conversation",
            ]
        ].to_dict(orient="records")
        if config.num_samples > 0:
            self.dataset = self.dataset[: config.num_samples]

    def run(self, strategy="token_probability", exclude_label="unsafe", **kwargs):
        if strategy in ["LogTokU", "probability_uncertainty", "entropy_uncertainty"]:
            return benchmark_logtoku(
                self.model,
                self.tokenizer,
                self.dataset,
                strategy=strategy,
                config=self.config,
                exclude_label=exclude_label,
            )
        processed_records = []
        if strategy == "greedy_generation":
            for record in tqdm(self.dataset, desc=f"Strategy - {strategy}"):
                probs = compute_probability_for_labels(
                    record["conversation"],
                    self.config,
                    self.model,
                    self.tokenizer,
                    strategy="token_probability",
                    top_p=kwargs.get("top_p", 0.99),
                    max_new_tokens=kwargs.get("max_new_tokens", 20),
                    eps=kwargs.get("eps", 1e-2),
                )
                pred_labels = [
                    label 
                    for label, _ in probs.items()
                    if label != exclude_label
                ]
                processed_records.append(
                    {
                        "ground_labels": record["ground_labels"],
                        "conversation": record["conversation"],
                        "pred_labels": pred_labels,
                    }
                )
            return processed_records
        for record in tqdm(self.dataset, desc=f"Strategy: {strategy}"):
            processed_records.append(
                {
                    "ground_labels": record["ground_labels"],
                    "conversation": record["conversation"],
                    "probs": compute_probability_for_labels(
                        record["conversation"],
                        self.config,
                        self.model,
                        self.tokenizer,
                        strategy=strategy,
                        top_p=kwargs.get("top_p", 0.99),
                        max_new_tokens=kwargs.get("max_new_tokens", 20),
                        eps=kwargs.get("eps", 1e-2),
                    ),
                }
            )
        return processed_records
