# 🛡️ Token-Level Marginalization for Multi-Label LLM Classifiers

This project explores methods for deriving interpretable confidence scores from generative large language models (LLMs) in the context of multi-label content safety classification. Generative models like LLaMA-Guard lack native support for class-level probabilities, which complicates thresholding and confidence-based decision-making. We propose and evaluate three token-level marginalization strategies to estimate class-wise probabilities from decoder outputs. Our findings demonstrate that these approaches improve interpretability, enable more fine-grained moderation control, and generalize well across instruction-tuned LLaMA-family models.

## 🚀 Quick Start

### 1. Install Dependencies

Make sure you have Python 3.11+ installed and a compatible NVIDIA GPU with CUDA support for GPU execution (if needed).

```bash
# Create a virtual environment using uv and Python 3.11
uv venv --python 3.11
source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Prepare Dataset

Ensure your dataset (e.g. `datasets/content_moderation_lm3.jsonl`) is available and formatted correctly (JSONL format, one sample per line).

### 3. Configure Parameters

All model- and strategy-specific parameters are defined in [`params.yaml`](./params.yaml).

```yaml
strategies:
  marginal_probability_at_any_place:
    top_p: 0.99
    max_new_tokens: 50
    eps: 0.01
  ...

models:
  meta-llama/Llama-3.1-8B-Instruct:
    top_p: 0.9   # Override strategy values globally for this model
```

### 4. Authentication (Hugging Face Token)

To access models hosted on Hugging Face (e.g., `meta-llama` models), you need to set your Hugging Face token as an environment variable. You can add the following to your `~/.bashrc` (or `~/.zshrc` if using zsh):

```bash
# Add your Hugging Face token
export HF_TOKEN=your_huggingface_token_here
```

Then, reload your shell configuration:

```bash
source ~/.bashrc  # or source ~/.zshrc
```


> 💡 Your token can be obtained from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 5. Run Evaluation

```bash
nohup bash -c "python main.py \
  --dataset datasets/content_moderation_lm3.jsonl \
  --model_id meta-llama/Llama-Guard-3-8B \
  --device cuda:0" > out.log 2>&1 &
```

This will:

* Load the model on the specified device
* Apply each strategy using the `params.yaml` configuration
* Log results to `out.log`
* Print and save a strategy-wise metrics table

---

## ⚙️ Parameters

| Parameter        | Description                                    |
| ---------------- | ---------------------------------------------- |
| `top_p`          | Sampling parameter for nucleus sampling        |
| `max_new_tokens` | Max tokens to generate per sample              |
| `eps`            | Epsilon used in probability estimation dfs     |
| `threshold`      | Used for calculating the final prediction      |

Model-specific overrides (e.g., `top_p`) take precedence over strategy defaults.

---

## 📁 Project Structure

```
.
├── datasets/
│   ├── content_moderation_lm2.jsonl
│   └── content_moderation_lm3.jsonl
├── src/
│   ├── utils.py
│   ├── config.py
│   ├── metrics.py
│   ├── methods.py
│   ├── runner.py
│   └── benchmark_logtoku.py
├── params.yaml
├── main.py
├── README.md
└── requirements.txt

```

---

## 📊 Output

The script outputs a DataFrame of metrics (e.g., accuracy, precision, recall) for each strategy and prints it to console. You can optionally export this to CSV by modifying the script:

```python
metrics_df.to_csv("results.csv")
```

---

## 🧠 Strategies Implemented

* `greedy_generation`
* `probability_uncertainty`
* `entropy_uncertainty`
* `LogTokU`
* `token_probability`
* `marginal_probability_at_fixed_place`
* `marginal_probability_at_any_place`

Each is evaluated independently with its configured parameters.

---

## 💡 Notes

* Use `bfloat16` for efficient inference on modern GPUs (e.g., A100, H100).
* If using a different tokenizer/model, adjust tokenizer loading accordingly.

---
