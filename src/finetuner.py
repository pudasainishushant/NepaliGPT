import numpy as np
import pandas as pd
from functools import partial
from data_loader import load_dataset_from_file
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_int8_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    set_seed,
    Trainer,
)

RESPONSE_KEY = " ### Response:"
DEFAULT_INPUT_MODEL = "Shushant/thesis_nepaliGPT"
seed = 42
model_checkpoint = "ftckpt"

class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        batch = super().torch_call(examples)
        response_token_ids = self.tokenizer.encode(RESPONSE_KEY, add_special_tokens=False)
        labels = batch["labels"].clone()
        for i in range(len(examples)):
            response_token_ids_start_idx = None
            for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                if np.array_equal(
                    response_token_ids,
                    batch["labels"][i, idx : idx + len(response_token_ids)],
                ):
                    response_token_ids_start_idx = idx
                    break
            if response_token_ids_start_idx is None:
                breakpoint()
                raise RuntimeError("Could not find response key token IDs")
            response_token_ids_end_idx = response_token_ids_start_idx + len(
                response_token_ids
            )
            labels[i, :response_token_ids_end_idx] = -100
        batch["labels"] = labels
        return batch


def preprocess_batch(batch, tokenizer: AutoTokenizer, max_length: int ):
    return tokenizer(batch["text"], max_length=max_length, truncation=True)


def load_training_dataset(
    training_data_id=load_dataset_from_file("datasets/finetuned_data/final_preprocessed.csv"),
):
    dataset = training_data_id
    data = load_dataset_from_file("datasets/finetuned_data/final_preprocessed.csv")
    data = data.filter(lambda rec: not rec["text"].strip().startswith(" ### Response:"))

    def _func(rec):
        rec["text"] += "\n\n### End"
        return rec

    dataset = dataset.map(_func)
    return dataset


def preprocess_dataset(
    tokenizer: AutoTokenizer, max_length: int, seed=seed, 
):
    dataset = load_training_dataset()
    _preprocessing_function = partial(
        preprocess_batch, max_length=max_length, tokenizer=tokenizer
    )
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["instruction", "input", "output", "text"],
    )
    dataset = dataset.filter(lambda rec: len(rec["input_ids"]) < max_length)
    dataset = dataset.shuffle(seed=seed)
    return dataset

def load_tokenizer(pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def load_model(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL,
    gradient_checkpointing: bool = False,
):
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        trust_remote_code=True,
        use_cache=False if gradient_checkpointing else True,
    )
    return model


def get_tokenizer_model(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL,
    gradient_checkpointing: bool = False,
):
    tokenizer = load_tokenizer(pretrained_model_name_or_path)
    model = load_model(
        pretrained_model_name_or_path, gradient_checkpointing=gradient_checkpointing
    )
    return tokenizer, model


def train(
    local_output_dir,
    epochs,
    per_device_train_batch_size,
    per_device_eval_batch_size,
    lr,
    seed,
    test_size=105,
):
    set_seed(seed)
    tokenizer, model = get_tokenizer_model()
    conf = model.config
    for length_setting in ['n_positions', 'max_position_embeddings', 'seq_length']:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            break
    if not max_length:
        max_length = 105

    processed_dataset = preprocess_dataset(tokenizer=tokenizer, max_length=max_length, seed=seed)
    split_dataset = processed_dataset.train_test_split(test_size=105, seed=seed)
    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
    )
    training_args = TrainingArguments(
        f"{model_checkpoint}-nepali-gpt",
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        optim = "adamw_torch",
        learning_rate=lr,
        weight_decay=0.01,
        gradient_checkpointing=True,
        fp16=True,
        warmup_ratio = 0.01,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        eval_steps=10,
        save_strategy="epoch",
        save_steps=200,
        save_total_limit=5,
        disable_tqdm=False,
        remove_unused_columns=True,
        lr_scheduler_type ='linear',
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(output_dir=local_output_dir)

def main(**kwargs):
    train(**kwargs)


if __name__ == "__main__":
    try:
        nep_tune = {
            "local_output_dir": "final-finetuned-nep-model",
            "epochs": 10,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "lr": 0.001,
            "seed": seed,
            "test_size": 105,
        }
        main(**nep_tune)
    except Exception:
        raise
