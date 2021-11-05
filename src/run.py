# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# https://cloud.tsinghua.edu.cn/u/d/6113ab997d104aa38110/
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import logging
import math
import os, sys
import random
import torch
import datasets
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.utils.versions import require_version
from collator import DataCollatorForLanguageModeling
from model import BertForMaskedLM
from trainer import Trainer

# label num for each task
dapt_tasks = {
    "chemprot": 13,
    "citation_intent": 6,
    "hyp": 2,
    "imdb": 2,
    "rct-20k": 5,
    "sciie": 7,
    "ag": 4,
    "amazon": 2,
}

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    # about active sampling
    parser.add_argument(
        "--tokenizer_file",
        type=str,
        default=None,
        help="The name of the tokenizer file"
    )
    parser.add_argument(
        "--external_ratio",
        type=int,
        default=0,
        help="the ratio of external data",
    )
    parser.add_argument(
        "--save_final",
        action="store_true",
        help="save the final checkpoint",
    )
    parser.add_argument(
        "--from_scratch",
        action="store_true",
        help="training from scratch",
    )
    parser.add_argument(
        "--reset_cls",
        action="store_true",
        help="reset the cls layer of the models",
    )
    parser.add_argument(
        "--from_ckpt",
        action="store_true",
        help="restore the model training process from a checkpoint",
    )
    parser.add_argument(
        "--mask_task",
        action="store_true",
        help="do random mask in downstream task dataset",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="The name of the directory storing the datasets"
    )
    parser.add_argument(
        "--external_dataset_name",
        type=str,
        default=None,
        help="The name of the dataset for mlm"
    )
    parser.add_argument(
        "--max_ckpts_to_keep",
        type=int,
        default=3,
        help="Number of checkpoints to keep"
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="Number of preprocessors"
    )
    parser.add_argument(
        "--steps_to_log",
        type=int,
        default=None,
        help="Num steps to log training info"
    )
    parser.add_argument(
        "--steps_to_eval",
        type=int,
        default=None,
        help="Num steps to evaluate on the dev set"
    )
    parser.add_argument(
        "--steps_to_save",
        type=int,
        default=None,
        help="Num steps to save the checkpoint"
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Number of preprocessors"
    )
    parser.add_argument(
        "--mlm_weight",
        type=float,
        default=20.0,
        help="the weight of mlm loss"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the task to train on.",
        choices=list(dapt_tasks.keys()),
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--max_external_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--disable_cls",
        action="store_true",
        help="disable the cls supervision signal.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default="./config/bert-base-uncased",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=None, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=0,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        # choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=10000, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--cuda_devices", type=str, default='0', help="visible cuda devices."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    args = parser.parse_args()
    # Sanity checks
    if args.task_name is None:
        raise ValueError("Need a task name.")
    
    if args.model_name_or_path is None:
        assert args.from_scratch, "no model name or path is provided but trying to initialize from a pre-trained weight"

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "args"), "w") as f:
            for arg in vars(args):
                f.write(f"{arg}: {getattr(args, arg)}\n")
    
    args.external_ratio += 1

    return args

def get_logger(args, accelerator):
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    if args.output_dir is not None:
        logfile = os.path.join(args.output_dir, "log")
        if accelerator.is_main_process:
            if os.path.exists(logfile):
                os.remove(logfile)
            os.mknod(logfile)
            fh = logging.FileHandler(logfile, mode='w')
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    logger.setLevel(logging.INFO if accelerator.is_main_process else logging.ERROR)
    if accelerator.is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    return logger

def get_external_dataset(args):
    data_files = {}
    if args.dataset_dir is not None:
        data_files["train"] = os.path.join(args.dataset_dir, args.task_name, args.external_dataset_name)
        extension = args.external_dataset_name.split(".")[-1]
        if extension == "txt":
                extension = "text"
        raw_datasets = load_dataset(extension, data_files=data_files)
    else:
        base_url = f"https://huggingface.co/datasets/yxchar/{args.task_name}-tlm/resolve/main/"
        data_files["train"] = base_url + args.external_dataset_name
        raw_datasets = load_dataset("csv", data_files=data_files)
    return raw_datasets

def preprocess_external(args, raw_datasets, tokenizer, logger):
    logger.info("preprocessing datasets")
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    padding = "max_length"
    def tokenize_function(examples):
        # Remove empty lines
        examples[text_column_name] = [
            line for line in examples[text_column_name] if line is not None and len(line) > 0 and not line.isspace()
        ]

        tokenized_examples = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=args.max_external_length,
            return_special_tokens_mask=True,
        )
        for k in examples:
            if k not in tokenized_examples and k != text_column_name and k != "rank" and k != "ids" and k != "id":
                tokenized_examples[k] = examples[k][:len(examples[text_column_name])]

        return tokenized_examples
    tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=[text_column_name],
            load_from_cache_file=True,
            desc="Running tokenizer on dataset line_by_line",
        )
    return tokenized_datasets["train"], text_column_name

def get_dataset(args):

    if args.dataset_dir is not None:
        task_dir = os.path.join(args.dataset_dir, args.task_name)
        data_files = {
            "train": os.path.join(task_dir, "train.csv"),
            "dev": os.path.join(task_dir, "dev.csv"),
            "test": os.path.join(task_dir, "test.csv"),
        }
        raw_datasets = load_dataset("csv", data_files=data_files)
    else:
        base_url = f"https://huggingface.co/datasets/yxchar/{args.task_name}-tlm/resolve/main/"
        data_files = {
            "train": base_url + "train.csv",
            "dev": base_url + "dev.csv",
            "test": base_url + "test.csv",
        }
        raw_datasets = load_dataset("csv", data_files=data_files)
    num_labels = dapt_tasks[args.task_name]
    label_list = list(range(num_labels))
    return raw_datasets, label_list, num_labels

def get_model(args, num_labels):
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(args.config_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.config_dir)

    if args.model_name_or_path and not args.from_scratch:
        set_cls = True if not args.from_scratch else False
        model = BertForMaskedLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            set_cls=set_cls,
            num_labels=num_labels,
        )
    else:
        model = BertForMaskedLM(config)
    if args.reset_cls or args.from_scratch:
        model.set_cls_layer(num_labels, config)
    model.set_args(args)    
    return tokenizer, model

def preprocess(args, model, tokenizer, raw_datasets, num_labels, label_list, logger, accelerator):
    padding = "max_length" if args.pad_to_max_length else False
    def preprocess_function(examples):
        # Tokenize the texts
        examples["text"] = [
            line for line in examples["text"] if line is not None and len(line) > 0 and not line.isspace()
        ]
        result = tokenizer(examples["text"], padding=padding, max_length=args.max_length, truncation=True, return_special_tokens_mask=True,)
        if "label" in examples:
            result["cls_labels"] = examples["label"]
        result.pop("rank", None)

        return result

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["dev"]
    test_dataset = processed_datasets["test"]
    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.mask_task:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    eval_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    return train_dataset, eval_dataset, test_dataset, data_collator, eval_data_collator

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    os.environ['NVIDIA_VISIBLE_DEVICES'] = args.cuda_devices
    
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    args.device = accelerator.device
    logger = get_logger(args, accelerator)
    raw_datasets, label_list, num_labels = get_dataset(args)
    tokenizer, model = get_model(args, num_labels)
    train_dataset, eval_dataset, test_dataset, data_collator, eval_data_collator = preprocess(args, model, tokenizer, raw_datasets, num_labels, label_list, logger, accelerator)
    
    if args.external_ratio > 1:
        raw_external_dataset = get_external_dataset(args)
        external_dataset, text_column_name = preprocess_external(args, raw_external_dataset, tokenizer, logger)
        external_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)
        external_dataloader = DataLoader(
            external_dataset, shuffle=True, collate_fn=external_data_collator, batch_size=args.per_device_train_batch_size
        )
    else:
        external_dataloader = None
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=eval_data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate,)

    # Prepare everything with `accelerator`.
    if external_dataloader is not None:
        model, optimizer, external_dataloader, train_dataloader = accelerator.prepare(
            model, optimizer, external_dataloader, train_dataloader
        )
    else:
        model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader
        )
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Get the metric function
    if args.task_name is not None:
        metric = load_metric("./src/metrics.py", args.task_name)
    else:
        metric = load_metric("accuracy")
    
    test_dataloader = DataLoader(
        test_dataset, collate_fn=eval_data_collator, batch_size=args.per_device_eval_batch_size
    )

    if args.model_name_or_path and args.from_ckpt:
        checkpoint_dir = args.model_name_or_path
    else:
        checkpoint_dir = None
    
    trainer = Trainer(
        args=args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        external_dataloader=external_dataloader,
        logger=logger,
        accelerator=accelerator,
        from_checkpoint=checkpoint_dir,
        test_dataloader=test_dataloader,
        metric=metric,
        label_list=label_list,
        tokenizer=tokenizer,
    )
    trainer.train()

if __name__ == "__main__":
    main()