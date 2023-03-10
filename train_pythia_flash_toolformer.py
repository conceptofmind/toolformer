# From: https://github.com/kyleliang919/Long-context-transformers
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import evaluate
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    GPTNeoXForCausalLM,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.models.gpt_neox.modeling_gpt_neox import RotaryEmbedding
from transformers.trainer_utils import get_last_checkpoint

from flash_attention.flash_attention_gptj_wrapper import FlashAttentionWrapper


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default="pythia-1.3b",
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )

    max_positions: Optional[int] = field(
        default=8192,
        metadata={"help": ("The maximum sequence length of the model.")},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="pile",
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    set_seed(training_args.seed)
    model = GPTNeoXForCausalLM.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.mask_token
    max_positions = model_args.max_positions
    tokenizer.model_max_length = max_positions
    for each in model.gpt_neox.layers:
        # original_emb = each.attention.rotary_emb
        each.attention.rotary_emb = RotaryEmbedding(
            each.attention.rotary_ndims, max_positions, 10000
        )
        each.attention.bias = torch.tril(
            torch.ones((max_positions, max_positions), dtype=torch.uint8)
        ).view(1, 1, max_positions, max_positions)
        each.attention = FlashAttentionWrapper(each.attention, max_seqlen=max_positions)

    # patching for the random contiguous tensors bug
    for p in model.parameters():
        p = p.contiguous()

    def merge_questions_and_answers(examples):
        out = tokenizer(
            [
                question + " " + answer
                for question, answer in zip(examples["input"], examples["output"])
            ]
        )
        return out

    block_size = tokenizer.model_max_length

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    if data_args.dataset_name == "pile":
        base_url = "https://the-eye.eu/public/AI/pile/"
        data_files = {
            "train": [
                base_url + "train/" + f"{idx:02d}.jsonl.zst" for idx in range(30)
            ],
            "validation": base_url + "val.jsonl.zst",
            "test": base_url + "test.jsonl.zst",
        }
        datasets = load_dataset("json", data_files=data_files, streaming=True)
        datasets = datasets.filter(lambda x: len(x["text"]) >= max_positions)
        tokenized_datasets = datasets.map(
            lambda examples: tokenizer(examples["text"]),
            batched=True,
        )
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
        )
        lm_datasets = lm_datasets.filter(lambda x: len(x["input_ids"]) >= max_positions)
    else:
        raise Exception("Sorry, please the dataset specified can not be recognized")

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    else:
        checkpoint = None
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    max_train_samples = len(train_dataset)
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
