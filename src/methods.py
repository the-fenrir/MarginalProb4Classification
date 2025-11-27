from collections import defaultdict

import torch
from transformers import GenerationConfig

from src.utils import get_input

def get_tokens_with_top_probabilities(logits, tokenizer, top_p=0.999):
    log_probs = torch.nn.functional.softmax(logits, dim=-1)
    sorted_indices = torch.argsort(log_probs, descending=True)
    sorted_probs = log_probs[sorted_indices]
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Get the cutoff index where cumulative prob first exceeds top_p
    cutoff_index = torch.searchsorted(cumulative_probs, top_p, right=True).item()

    # Select top tokens up to and including cutoff_index
    top_indices = sorted_indices[: cutoff_index + 1]
    top_probs = sorted_probs[: cutoff_index + 1]
    top_logits = logits[top_indices]

    top_tokens = tokenizer.batch_decode(top_indices)
    return dict(zip(top_tokens, zip(top_probs.tolist(), top_indices.tolist())))

def _compute_token_probability(
    inputs,
    labels,
    model,
    tokenizer,
    max_new_tokens=20,
    top_p=0.999,
    generation_config=None,
    device="cuda:0",
):
    probabilities = defaultdict(float)

    outputs = model.generate(
        **inputs,
        generation_config=generation_config,
        max_new_tokens=max_new_tokens,
    )

    for logits in outputs.logits:
        top_tokens = get_tokens_with_top_probabilities(logits[0], tokenizer, top_p=top_p)
        for idx, (token, (prob, token_id)) in enumerate(top_tokens.items()):
            inputs = {
                **inputs,
                "input_ids": torch.cat(
                    [inputs["input_ids"], torch.tensor([[token_id]]).to(device)], dim=1
                ),
                "attention_mask": torch.cat(
                    [inputs["attention_mask"], torch.tensor([[1]]).to(device)], dim=1
                ),
            }
            generation = tokenizer.batch_decode(inputs["input_ids"])[0]
            for label in labels:
                if generation.endswith(label):
                    probabilities[label] = prob
                    break
            break
    return probabilities

def _compute_marginal_probability_at_fixed_place(
    inputs,
    labels,
    model,
    tokenizer,
    max_new_tokens=20,
    top_p=0.999,
    generation_config=None,
    device="cuda:0",
):
    probabilities = defaultdict(float)

    outputs = model.generate(
        **inputs,
        generation_config=generation_config,
        max_new_tokens=max_new_tokens,
    )

    curr_prob = 1
    for logits in outputs.logits:
        top_tokens = get_tokens_with_top_probabilities(logits[0], tokenizer, top_p=top_p)
        for idx, (token, (prob, token_id)) in enumerate(top_tokens.items()):
            inputs = {
                **inputs,
                "input_ids": torch.cat(
                    [inputs["input_ids"], torch.tensor([[token_id]]).to(device)], dim=1
                ),
                "attention_mask": torch.cat(
                    [inputs["attention_mask"], torch.tensor([[1]]).to(device)], dim=1
                ),
            }
            generation = tokenizer.batch_decode(inputs["input_ids"])[0]
            for label in labels:
                if generation.endswith(label):
                    probabilities[label] = curr_prob * prob
                    curr_prob *= prob
                    break
            break
    return probabilities

def _compute_marginal_probability_at_any_place(
    inputs,
    labels,
    model,
    tokenizer,
    eos_token_ids,
    max_new_tokens=10,
    top_p=0.999,
    eps=1e-7,
    generation_config=None,
    device="cuda:0",
):
    probabilities = defaultdict(float)

    def dfs(inputs, labels, curr_prob=1, depth=0, max_new_tokens=10, device="cuda:0"):
        if curr_prob < eps:
            return
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
            max_new_tokens=1,
        )
        top_tokens = get_tokens_with_top_probabilities(
            outputs.logits[0][0], tokenizer, top_p=top_p
        )
        top_token_ids = [token_id for _, (prob, token_id) in top_tokens.items()]
        for idx, (token, (prob, token_id)) in enumerate(top_tokens.items()):
            new_inputs = {
                **inputs,
                "input_ids": torch.cat(
                    [inputs["input_ids"], torch.tensor([[token_id]]).to(device)], dim=1
                ),
                "attention_mask": torch.cat(
                    [inputs["attention_mask"], torch.tensor([[1]]).to(device)], dim=1
                ),
            }
            generation = tokenizer.batch_decode(new_inputs["input_ids"])[0]
            for label in labels:
                if generation.endswith(label):
                    probabilities[label] += curr_prob * prob
            if token_id in eos_token_ids and prob >= 0.7:
                break
            if (
                any(token_id in top_token_ids for token_id in eos_token_ids)
                and idx == 2
            ):
                break
            if depth == max_new_tokens or token_id in eos_token_ids:
                continue
            dfs(new_inputs, labels, curr_prob * prob, depth + 1, max_new_tokens, device)

    dfs(inputs, labels, max_new_tokens=max_new_tokens, device=device)
    return probabilities

def compute_probability_for_labels(
    conversation,
    config,
    model,
    tokenizer,
    strategy="marginal_probability_at_any_place",
    max_new_tokens=10,
    top_p=0.999,
    eps=1e-7,
):
    inputs = get_input(
        conversation,
        config,
        tokenizer,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
        add_generation_prompt=True,
    )
    inputs = {k: v.to(config.device) for k, v in inputs.items()}

    generation_config = GenerationConfig(
        max_new_tokens=10,
        do_sample=False,
        temperature=1,
        decoder_start_token_id=0,
        pad_token_id=config.eos_token_ids[0],
        output_scores=True,
        output_logits=True,
        return_dict_in_generate=True,
    )
    if strategy == "token_probability":
        return _compute_token_probability(
            inputs,
            config.labels,
            model,
            tokenizer,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            generation_config=generation_config,
            device=config.device,
        )
    elif strategy == "marginal_probability_at_fixed_place":
        return _compute_marginal_probability_at_fixed_place(
            inputs,
            config.labels,
            model,
            tokenizer,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            generation_config=generation_config,
            device=config.device,
        )
    elif strategy == "marginal_probability_at_any_place":
        return _compute_marginal_probability_at_any_place(
            inputs,
            config.labels,
            model,
            tokenizer,
            eos_token_ids=config.eos_token_ids,
            top_p=top_p,
            eps=eps,
            max_new_tokens=max_new_tokens,
            generation_config=generation_config,
            device=config.device,
        )