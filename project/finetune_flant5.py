import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import time
import argparse
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer, BitsAndBytesConfig
import torch
import evaluate
import pandas as pd
import numpy as np
import json
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig, prepare_model_for_kbit_training


def arg():
    args = argparse.ArgumentParser()
    args.add_argument("--full_fine_tune", action=argparse.BooleanOptionalAction, default=False)
    args.add_argument("--local_full_tune", action=argparse.BooleanOptionalAction, default=False)
    args.add_argument("--r", type=int, default=8)
    args.add_argument("--alpha", type=int, default=16)
    args.add_argument("--target_modules", nargs="+", default=["q","v"])
    args.add_argument("--lr",type=float,default=1e-3)
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
        max_steps=steps)

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
         ,tokenized_train_data
         ,tokenized_val_data
         ,r
         ,alpha
         ,target_modules
         ,lr
         ,epoch
         ,steps):
    # 1. build peft adaptor
    parameters = {"r":r,
                  "lora_alpha":alpha,
                  "target_modules":target_modules}
    
    lora_config = LoraConfig(
        **parameters,
        lora_dropout= 0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )


    peft_model = get_peft_model(base_model, lora_config)

    # 2. train
    training_args = TrainingArguments(
        output_dir=f'./runs/peft-dialogue-summary-training-{str(int(time.time()))}',
        auto_find_batch_size=True,
        learning_rate=lr,
        num_train_epochs=epoch,
        logging_steps=1,
        max_steps=steps,
        eval_strategy="steps",
        eval_steps=10,
        load_best_model_at_end=True
    )

    peft_trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_val_data
    )

    peft_trainer.train()
    peft_trainer.model.save_pretrained(store_model_path)


def build_run_metadata(arguments):
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "full_fine_tune" if arguments.full_fine_tune else "qlora",
        "hyperparameters": {
            "r": arguments.r,
            "alpha": arguments.alpha,
            "target_modules": arguments.target_modules,
            "lr": arguments.lr,
            "epoch": arguments.epoch,
            "weight_decay": arguments.weight_decay,
            "steps": arguments.steps,
        },
    }


def evaluate_rouge(model_name
                   ,pretrained_peft_model_path
                   ,pretrained_instruct_model_path  # "./flan-dialogue-summary-checkpoint"
                   ,tokenizer
                   ,dataset
                   ,run_metadata
                   ,output_dir="."):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    base_model_summaries = []
    peft_model_summaries = []
    instruct_model_summaries = []

    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name,
                                                    dtype=torch.float16).to(device)

    peft_base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name,
                                                    dtype=torch.float16).to(device)
    peftuned_model = PeftModel.from_pretrained(peft_base_model, pretrained_peft_model_path).to(device)

    instruct_model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_instruct_model_path, 
                                                           dtype=torch.float16).to(device)

    # completion_path = os.path.join(output_dir, "test_completion.jsonl")
    # with open(completion_path, "w", encoding="utf-8") as f:

        # for index in range(dataset["test"].shape[0]):
        #     print(f"Evaluating {index + 1}/{dataset['test'].shape[0]}")
        #     prompt = f"""
        #     Summarize the following conversation.

        #     {dataset["test"][index]["dialogue"]}

        #     Summary: """

        #     input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        #     base_model_outputs = base_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
        #     base_model_text_output = tokenizer.decode(base_model_outputs[0], skip_special_tokens=True)
        #     base_model_summaries.append(base_model_text_output)

        #     instruct_model_outputs = instruct_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
        #     instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)
        #     instruct_model_summaries.append(instruct_model_text_output)

        #     peft_model_outputs = peftuned_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
        #     peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)
        #     peft_model_summaries.append(peft_model_text_output)

        #     print('###', index)
        #     print('/nORIGINAL MODEL:',base_model_text_output)
        #     print('/nPEFT MODEL:', peft_model_text_output)
        #     print('/nINSTRUCT MODEL:', instruct_model_text_output)

        #     output =  {
        #         'BASELINE HUMAN SUMMARY':dataset["test"][index]["summary"],
        #         'ORIGINAL MODEL':base_model_text_output,
        #         'PEFT MODEL': peft_model_text_output,
        #         'INSTRUCT MODEL': instruct_model_text_output
        #     }
            
            # f.write(json.dumps(output, ensure_ascii=False) + "\n")
    
    test_data = dataset["test"]
    n = test_data.shape[0]
    prompts = [
        "Summarize the following conversation.\n"
        + test_data[i]["dialogue"]
        + "\nSummary:"
        for i in range(n)
    ]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True).to(device)

    gen_config = GenerationConfig(max_new_tokens=200, num_beams=1)

    t0 = time.time()
    base_outputs = base_model.generate(**inputs, generation_config=gen_config)
    print(f"base generate took {time.time()-t0:.2f}s")

    t0 = time.time()
    instruct_outputs = instruct_model.generate(**inputs, generation_config=gen_config)
    print(f"instruct generate took {time.time()-t0:.2f}s")

    t0 = time.time()
    peft_outputs = peftuned_model.generate(**inputs, generation_config=gen_config)
    print(f"peft generate took {time.time()-t0:.2f}s")

    base_model_summaries = tokenizer.batch_decode(base_outputs, skip_special_tokens=True)
    instruct_model_summaries = tokenizer.batch_decode(instruct_outputs, skip_special_tokens=True)
    peft_model_summaries = tokenizer.batch_decode(peft_outputs, skip_special_tokens=True)

    completion_path = os.path.join(output_dir, "test_completion.jsonl")
    with open(completion_path, "w", encoding="utf-8") as f:
        for i in range(n):
            output = {
                'BASELINE HUMAN SUMMARY': test_data[i]["summary"],
                'ORIGINAL MODEL': base_model_summaries[i],
                'PEFT MODEL': peft_model_summaries[i],
                'INSTRUCT MODEL': instruct_model_summaries[i],
            }
            f.write(json.dumps(output, ensure_ascii=False) + "\n")

    rouge = evaluate.load('rouge')

    base_model_results = rouge.compute(
        predictions=base_model_summaries,
        references=dataset["test"]["summary"],#[0:len(base_model_summaries)],
        use_aggregator=True,
        use_stemmer=True,
    )

    peft_model_results = rouge.compute(
        predictions=peft_model_summaries,
        references=dataset["test"]["summary"],#[0:len(peft_model_summaries)],
        use_aggregator=True,
        use_stemmer=True,
    )

    instruct_model_results = rouge.compute(
            predictions=instruct_model_summaries,
            references=dataset["test"]["summary"],#[0:len(peft_model_summaries)],
            use_aggregator=True,
            use_stemmer=True,
        )
    run_metadata["rouge"] = {
        "base_model": base_model_results,
        "peft_model": peft_model_results,
        "instruct_model": instruct_model_results,
    }
    metadata_path = os.path.join(output_dir, f"run_metadata_{dataset['train'].shape[0]}.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, ensure_ascii=False, indent=2)

    return base_model_results, peft_model_results, instruct_model_results



if __name__ == "__main__":
    arguments = arg()

    dataset = load_dataset("knkarthick/dialogsum")
    # dataset = dataset.filter(lambda example, index: index % 100 == 0, with_indices=True)
    dataset = DatasetDict({
        "train":dataset["train"].shuffle(seed=42).select(range(1000)),
        "validation": dataset["validation"].shuffle(seed=42).select(range(100)),
        # "validation": dataset["validation"].shuffle(seed=42).select(range(25)),  
        "test": dataset["test"].filter(lambda example, index: index % 50 == 0,with_indices=True)
    })

    # load tokenizer
    model_name = 'google/flan-t5-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Preprocess dataset
    tokenized_datasets = dataset.map(tokenize_function, 
                                     batched=True,
                                     fn_kwargs={"tokenizer": tokenizer})
    
    tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary',])
    print(tokenized_datasets.shape)

    run_dir = f"./results/{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(run_dir, exist_ok=True)

    # peft - lora
    if not arguments.full_fine_tune:
        peft_path = "./results/peft-dialogue-summary-checkpoint-local" 
        # peft_path = "./peft-dialogue-summary-training-1790834018/checkpoint-126"
        instruct_path = "./results/flan-dialogue-summary-checkpoint-aws"

        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=torch.float16)

        # qlora:###
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            attn_implementation="eager")
        
        base_model = prepare_model_for_kbit_training(base_model)
        ####

        peft_adaptor(base_model=base_model
            ,store_model_path=peft_path
            ,tokenized_train_data=tokenized_datasets["train"]
            ,tokenized_val_data=tokenized_datasets["validation"]
            ,r=arguments.r
            ,alpha=arguments.alpha
            ,target_modules=arguments.target_modules
            ,lr=arguments.lr
            ,epoch=arguments.epoch
            ,steps=arguments.steps)

        print("Tuning done.")

    else: # full fine tune the model
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=torch.float32)
        peft_path = "./runs/peft-dialogue-summary-checkpoint-local"
        
        if arguments.local_full_tune:
            instruct_path = "./runs/flan-dialogue-summary-checkpoint-local"
            full_fine_tune(store_full_model_path = instruct_path
                ,base_model=base_model
                ,tokenized_train_data=tokenized_datasets["train"]
                ,tokenized_val_data=tokenized_datasets["validation"]
                ,lr=arguments.lr
                ,weight_decay=arguments.weight_decay
                ,epoch=arguments.epoch
                ,steps=arguments.steps)

        else:
            instruct_path = "./runs/flan-dialogue-summary-checkpoint-aws"

    run_metadata = build_run_metadata(arguments)

    base_results, peft_results, instruct_results = evaluate_rouge(
        model_name
        ,pretrained_peft_model_path=peft_path
        ,pretrained_instruct_model_path=instruct_path
        ,tokenizer=tokenizer
        ,dataset=dataset
        ,run_metadata=run_metadata
        ,output_dir=run_dir)

    print("Base model ROUGE:")
    print(base_results)

    print("PEFT model ROUGE:")
    print(peft_results)

    print("INSTRUCT model ROUGE:")
    print(instruct_results)