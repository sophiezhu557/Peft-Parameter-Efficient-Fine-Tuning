# Parameter-Efficient Fine-Tuning (PEFT)

This project explores Parameter-Efficient Fine-Tuning (PEFT) techniques for dialogue summarization, comparing pretrained models with fine-tuned models in terms of summarization quality and factual accuracy.

## Project Structure

- `finetune.py` — Fine-tuning and evaluation using **FLAN-T5**.
- `finetune_v2.py` — Fine-tuning and evaluation using **Qwen**.
- `requirements.txt` — Python dependencies.

## Dataset

[DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum) — A dataset of real-life dialogues paired with human-written summaries.

## Methods

- **Baseline:** Zero-shot and few-shot prompting.
- **PEFT:** LoRA / QLoRA for efficient fine-tuning.
- **Full Fine-Tuning:** Used for model performance comparison.

## Evaluation

Model performance is evaluated using:

- **ROUGE scores** — Measuring similarity to reference summaries.
- **Factual accuracy** — Checking whether generated summaries correctly preserve key information and speaker–fact relationships.

## Tech Stack

Python · PyTorch · Hugging Face Transformers · PEFT · Datasets · Evaluate

## Installation

```bash
pip install -r requirements.txt
```

## Objective

To understand how different fine-tuning techniques and model architectures affect the efficiency and accuracy of dialogue summarization.
