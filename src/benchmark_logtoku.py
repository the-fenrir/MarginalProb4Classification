import numpy as np
import pandas as pd

import torch
from scipy.special import digamma, softmax
from src.utils import get_input
from tqdm import tqdm
from transformers import GenerationConfig


def compute_logits_for_labels(
    conversation,
    config,
    model,
    tokenizer,
    label_to_idx,
    add_half_generation=False,
):
    inputs = get_input(
        conversation,
        config,
        tokenizer,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
        add_generation_prompt=True,
        add_initial_generation=True,
        add_half_generation=add_half_generation,
    )
    inputs = {k: v.to(config.device) for k, v in inputs.items()}

    generation_config = GenerationConfig(
        max_new_tokens=1,
        do_sample=False,
        temperature=1,
        decoder_start_token_id=0,
        pad_token_id=config.eos_token_ids[0],
        output_scores=True,
        output_logits=True,
        return_dict_in_generate=True,
    )
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=generation_config)

        label_logits = {
            l: outputs.scores[0][0, tid].item() for l, tid in label_to_idx.items()
        }

        return label_logits


def topk(arr, k):
    indices = np.argpartition(arr, -k)[-k:]
    values = arr[indices]
    return values, indices


def process_data(dataset):
    processed_dataset = []
    for x in dataset:
        label_vector = np.array(
            [x["label_vector"][label] for label in sorted(x["label_vector"])]
        )
        label_logits = np.array(
            [x["label_logits"][label] for label in sorted(x["label_logits"])]
        )
        label_prob = softmax(label_logits)

        processed_dataset.append(
            {
                "ground_labels": x["ground_labels"],
                "conversation": x["conversation"],
                "label_vector": label_vector,
                "label_logits": label_logits,
                "label_prob": label_prob,
            }
        )
    return processed_dataset


def get_u(mode, k=None):
    if mode == "probability_uncertainty":

        def u(logits):
            logits = softmax(logits)
            top_k = 1
            if len(logits) < top_k:
                raise ValueError("Logits array length is less than top_k.")
            top_values, _ = topk(logits, top_k)
            mean_scores = top_k / (np.sum(np.maximum(0, top_values)) + top_k)
            return mean_scores

        return u

    elif mode == "entropy_uncertainty":

        def u(logits):
            probs = softmax(logits)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            return entropy

        return u

    elif mode == "LogTokU":

        def cal_au(alpha):
            alpha = np.array([alpha])
            alpha_0 = alpha.sum(axis=1, keepdims=True)
            psi_alpha_k_plus_1 = digamma(alpha + 1)
            psi_alpha_0_plus_1 = digamma(alpha_0 + 1)
            result = -(alpha / alpha_0) * (psi_alpha_k_plus_1 - psi_alpha_0_plus_1)
            result = result.sum(axis=1)
            return result

        def u(logits):
            top_k = k
            if len(logits) < top_k:
                raise ValueError("Logits array length is less than top_k.")
            top_values = np.partition(logits, -top_k)[-top_k:]
            au = cal_au(top_values)
            mean_scores = top_k / (np.sum(np.maximum(0, top_values)) + top_k)
            # return mean_scores
            return mean_scores / au

        return u

    else:
        raise ValueError(f"Unsupported mode: {mode}")


def CalSecond(label_list, pred_sec, accumu):
    first_indices = pred_sec[0]
    second_indices = pred_sec[1]
    if label_list[first_indices] and label_list[second_indices]:
        accumu += 1
    elif label_list[first_indices] and not label_list[second_indices]:
        accumu -= 1
    elif not label_list[first_indices] and label_list[second_indices]:
        accumu -= 0
    else:
        accumu -= 0
    return accumu


def Monopoly(label_list, pred_list):
    game_score = 0
    for pred in pred_list:
        if label_list[pred]:
            game_score += 1
        else:
            game_score -= 1
    return max(0, game_score)


def sort_processed_data_by_total_u(processed_data, u):
    """Calculate the u value for each entry and store it together with the original entry in a list."""
    data_with_u = [(entry, u(entry["label_logits"])) for entry in processed_data]
    u_values = [u_value for _, u_value in data_with_u]
    sorted_data_with_u = sorted(data_with_u, key=lambda x: x[1])
    sorted_processed_data = [entry for entry, u_value in sorted_data_with_u]
    return sorted_processed_data, u_values


def dynamic_select(logits, vector, sorted_indices, t_opti, u):
    sample_u = u(logits)

    if sample_u > t_opti:
        choice = 1
    else:
        choice = 2
    return Monopoly(vector, sorted_indices[:choice]), choice


def calculate_t_opti(sorted_processed_data, u):
    accumu_list = []
    accumu = 0
    for entry in sorted_processed_data:
        # Get sorted indices (ascending order), reverse for descending order
        sorted_indices = np.argsort(entry["label_logits"])[::-1]
        # Update accumu with the result of CalSecond
        accumu = CalSecond(entry["label_vector"], sorted_indices[:2], accumu)

        accumu_list.append(accumu)

    # Find the max value and its index in accumu_list
    max_value = max(accumu_list)
    max_idx = accumu_list.index(max_value)
    # Calculate and return the t_opti value
    t_opti = u(sorted_processed_data[max_idx]["label_logits"])
    return t_opti


def benchmark_logtoku(
    model, tokenizer, dataset, strategy="LogTokU", config=None, exclude_label="unsafe"
):
    labels = [label for label in config.labels if label != exclude_label]
    label_to_idx = {e: tokenizer.encode(e, add_special_tokens=False)[1] for e in labels}
    safety_labels_to_idx = {
        e: tokenizer.encode(e, add_special_tokens=False)[0] for e in ["safe", "unsafe"]
    }

    is_unsafe = [True] * len(dataset)
    if config.dataset == "beavertails":
        is_unsafe = []
        for data in tqdm(dataset, desc="Beavertails - Safety Check"):
            safety_logits = compute_logits_for_labels(
                data["conversation"],
                config,
                model,
                tokenizer,
                safety_labels_to_idx,
                add_half_generation=True,
            )
            is_unsafe.append(safety_logits["unsafe"] > safety_logits["safe"])

    dataset = pd.DataFrame(dataset)
    filtered_df = dataset[is_unsafe].reset_index(drop=True)

    label_vector_list = []
    label_logits_list = []

    for i, record in tqdm(filtered_df.iterrows(), desc=f"Strategy - {strategy}"):

        label_vector_list.append(
            {l: l in record["ground_labels"] for l in label_to_idx}
        )
        label_logits_list.append(
            compute_logits_for_labels(
                record["conversation"],
                config,
                model,
                tokenizer,
                label_to_idx,
            )
        )

    filtered_df["label_vector"] = label_vector_list
    filtered_df["label_logits"] = label_logits_list
    filtered_df = filtered_df[
        ["ground_labels", "conversation", "label_vector", "label_logits"]
    ]

    processed_data = process_data(filtered_df.to_dict("records"))

    sorted_labels = sorted(labels)

    mode_k_dict = {
        "probability_uncertainty": None,
        "entropy_uncertainty": None,
        "LogTokU": 2,
    }

    u_function = get_u(strategy, mode_k_dict[strategy])
    sorted_processed_data, u_values = sort_processed_data_by_total_u(
        processed_data, u_function
    )
    t_opti = calculate_t_opti(sorted_processed_data, u_function)

    for entry in processed_data:
        sorted_indices = np.argsort(entry["label_logits"])[::-1]
        score, num_choices = dynamic_select(
            entry["label_logits"],
            entry["label_vector"],
            sorted_indices,
            t_opti,
            u_function,
        )
        entry["pred_labels"] = [
            sorted_labels[id] for id in sorted_indices[:num_choices]
        ]

    dataset["pred_labels"] = [[]] * len(dataset)
    indices = dataset[is_unsafe].index
    for i, idx in enumerate(indices):
        dataset.at[idx, "pred_labels"] = processed_data[i]["pred_labels"]

    return dataset[["ground_labels", "conversation", "pred_labels"]].to_dict("records")