import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import time
import argparse
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer, BitsAndBytesConfig
import torch
import evaluate
import pandas as pd
import numpy as np
import json
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig, prepare_model_for_kbit_training
from transformers import GenerationConfig 
from trl import SFTTrainer, SFTConfig


def arg():
    args = argparse.ArgumentParser()
    args.add_argument("--r", type=int, default=8)
    args.add_argument("--alpha", type=int, default=16)
    args.add_argument("--lr",type=float,default=None)
    args.add_argument("--epoch", type=int, default=1)
    args.add_argument("--weight_decay", type=float, default=0.01)
    args.add_argument("--steps", type=int, default=-1)
    return args.parse_args()


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, params in model.named_parameters():
        all_model_params += params.numel()
        if params.requires_grad:
            trainable_model_params += params.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"


def tokenize_function(example, tokenizer):
    prompt = ["Summarize the following conversation. \n" 
            + dialogue 
            + "\nSummary:" 
            for dialogue in example["dialogue"]]

    inputs = tokenizer(
        prompt, 
        padding = "max_length", 
        truncation=True)
    targets = tokenizer(
        example["summary"], 
        padding="max_length", 
        truncation=True)

    labels = [
        [token if token != tokenizer.pad_token_id else -100 for token in label]
        for label in targets["input_ids"]
    ]
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels,
    }


def full_fine_tune(store_full_model_path
                    ,base_model
                    ,tokenized_train_data
                    ,tokenized_val_data
                    ,lr
                    ,weight_decay
                    ,epoch
                    ,steps):
    training_args = TrainingArguments(
        output_dir=f'./dialogue-summary-training-{str(int(time.time()))}',
        # auto_find_batch_size=True,
        learning_rate=lr,
        num_train_epochs=epoch,
        weight_decay=weight_decay,
        logging_steps=1,
        max_steps=steps,
        gradient_checkpointing=False)

    full_trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_val_data
    )

    batch = next(iter(full_trainer.get_train_dataloader()))

    print("input dtype:", batch["input_ids"].dtype)
    print("label dtype:", batch["labels"].dtype)

    print(
        "valid labels:",
        (batch["labels"] != -100).sum().item()
    )

    print(
        "ignored labels:",
        (batch["labels"] == -100).sum().item()
    )

    full_trainer.train()
    full_trainer.model.save_pretrained(store_full_model_path)


def peft_adaptor(store_model_path
         ,base_model
         ,tokenizer
         ,dataset
         ,r
         ,alpha
         ,target_modules
         ,lr
         ,epoch
         ,steps):
    parameters = {"r":r,
                  "lora_alpha":alpha,
                  "target_modules":target_modules}
    lora_config = LoraConfig(
        **parameters,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM 
    )


    peft_model = get_peft_model(base_model, lora_config)

    training_args = SFTConfig(
        output_dir=f'./runs/peft-dialogue-summary-training-{str(int(time.time()))}',
        learning_rate=lr,
        num_train_epochs=epoch,
        max_steps=steps,
        completion_only_loss=True,
    )

    peft_trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"]
    )

    peft_trainer.train()
    peft_trainer.model.save_pretrained(store_model_path)
    return peft_trainer.model


def build_run_metadata(arguments):
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "qlora",
        "hyperparameters": {
            "r": arguments.r,
            "alpha": arguments.alpha,
            "target_modules": target_modules,
            "lr": arguments.lr,
            "epoch": arguments.epoch,
            "weight_decay": arguments.weight_decay,
            "steps": arguments.steps,
        },
    }


def evaluate_rouge(model_name
                   ,pretrained_peft_model_path
                   ,tokenizer
                   ,dataset
                   ,run_metadata
                   ,output_dir="."
                   ,max_new_tokens=80
                   ,debug_first_n=2):
    
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[eval] device={device}")

    eval_dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[eval] eval_dtype={eval_dtype}")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=eval_dtype, attn_implementation="eager").to(device)   
    peft_model = PeftModel.from_pretrained(base_model, pretrained_peft_model_path).to(device)
    peft_model.eval()

    base_summaries, peft_summaries = [], []
    completion_path = os.path.join(output_dir, "test_completion.jsonl")
    with open(completion_path, "w", encoding="utf-8") as f:
        for i, example in enumerate(dataset["test"]):
            prompt = ("Summarize the following conversation.\n" + example["dialogue"] + "\nSummary:")
            messages = [
                {"role": "user", "content": prompt}
            ]
            inputs = tokenizer.apply_chat_template(messages,
                                                    tokenize=True,
                                                    add_generation_prompt=True,
                                                    return_dict=True,
                                                    return_tensors="pt"
                                                ).to(device)
            
            with torch.no_grad():
                with peft_model.disable_adapter():
                    base_output = peft_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                peft_output = peft_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

            if i < debug_first_n:
                in_len = inputs["input_ids"].shape[1]
                print(f"[debug] input_len={in_len} "
                      f"base_out_shape={tuple(base_output.shape)} "
                      f"peft_out_shape={tuple(peft_output.shape)}")
                print("[debug] base new ids:", base_output[0][in_len:in_len+20].tolist())
                print("[debug] peft new ids:", peft_output[0][in_len:in_len+20].tolist())

            input_length = inputs["input_ids"].shape[1]
            base_summary = tokenizer.decode(
                base_output[0, input_length:],
                skip_special_tokens=True,
            )
            peft_summary = tokenizer.decode(
                peft_output[0, input_length:],
                skip_special_tokens=True,
            )
            
            base_summaries.append(base_summary)
            peft_summaries.append(peft_summary)
            output = {
                "dialogue": example["dialogue"],
                "BASELINE HUMAN SUMMARY": example["summary"],
                "ORIGINAL MODEL": base_summary,
                "PEFT MODEL": peft_summary,
            }
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
            f.flush()
            print(f"Evaluating {i + 1}/{len(dataset['test'])}")

    # ROUGE
    rouge = evaluate.load("rouge")
    predictions = {"base_model": base_summaries, "peft_model": peft_summaries}
    results = {}
    for name, summaries in predictions.items():
        results[name] = rouge.compute(
            predictions=summaries,
            references=list(dataset["test"]["summary"]),
            use_stemmer=True,
        )
    run_metadata["model_name"] = model_name
    run_metadata["rouge"] = results
    with open(os.path.join(output_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, ensure_ascii=False, indent=2)
    return results["base_model"], results["peft_model"]


def prepare_prompt(example):
    return {
        "prompt": [
            {
                "role": "user",
                "content": (
                    "Summarize the following conversation.\n"
                    + example["dialogue"]
                    + "\nSummary:"
                ),
            }
        ],
        "completion": [
            {
                "role": "assistant",
                "content": example["summary"],
            }
        ],
    }

    

if __name__ == "__main__":
    arguments = arg()

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"  #Qwen/Qwen3.5-2B"
    peft_path = "./runs/peft-dialogue-summary-checkpoint-local-qwen2.5"
    target_modules = ["q_proj", "v_proj"]
    lr = 1e-4

    instruct_path = "./flan-dialogue-summary-checkpoint-aws"
    arguments.lr = arguments.lr if arguments.lr is not None else lr

    dataset = load_dataset("knkarthick/dialogsum")
    dataset = DatasetDict({
        "train": dataset["train"].shuffle(seed=42).select(range(500)),
        "validation": dataset["validation"].shuffle(seed=42).select(range(50)),
        "test": dataset["test"].filter(
            lambda example, index: index % 50 == 0, with_indices=True),
    })
    run_dir = f"./results/{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(run_dir, exist_ok=True)

    processerd_dataset = dataset.map(
        prepare_prompt, remove_columns=dataset["train"].column_names)

    print(processerd_dataset.shape)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config,
        attn_implementation="eager")

    base_model = prepare_model_for_kbit_training(base_model)
    base_model.config.use_cache = False


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    peft_model = peft_adaptor(
        store_model_path=peft_path,
        base_model=base_model,
        tokenizer=tokenizer,
        dataset=processerd_dataset,
        r=arguments.r,
        alpha=arguments.alpha,
        target_modules=target_modules,
        lr=arguments.lr,
        epoch=arguments.epoch,
        steps=arguments.steps,
    )
    tokenizer.save_pretrained(peft_path)
    del peft_model
    print("Tuning done.")

    # 评估前释放训练模型，避免重复占用内存。
    del base_model
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    run_metadata = build_run_metadata(arguments)
    base_results, peft_results = evaluate_rouge(
        model_name=model_name,
        pretrained_peft_model_path=peft_path,
        tokenizer=tokenizer,
        dataset=dataset,
        run_metadata=run_metadata,
        output_dir=run_dir,
    )
    print("Base model ROUGE:", base_results)
    print("PEFT model ROUGE:", peft_results)

