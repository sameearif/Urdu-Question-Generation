# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 12:40:14 2021

@author: ozcan
"""
from lib2to3.pgen2.tokenize import tokenize
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import pandas as pd

from nlp import Dataset
from transformers import MT5Tokenizer, HfArgumentParser
from process_data import generate_data


logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pretaining to what data we are going to input our model for training and eval.
    """
    input_path: Optional[str] = field(
        default="",
        metadata={"help": "Path for json dataset directory"}, 
    )
    output_path: Optional[str] = field(
        default="data",
        metadata={"help": "Path for output dataset directory"}, 
    )
    tokenizer_path: Optional[str] = field(
        default="mt5_qg_tokenizer",
        metadata={"help": "Path for ttokenizer"}, 
    )
    input_train_file: Optional[str] = field(
        default="train.json",
        metadata={"help": "input train json file"}, 
    )
    input_eval_file: Optional[str] = field(
        default="eval.json",
        metadata={"help": "input train json file"}, 
    )
    output_train_file: Optional[str] = field(
        default=None,
        metadata={"help": "name for cached train dataset"},
    )
    output_eval_file: Optional[str] = field(
        default=None,
        metadata={"help": "name for cached valid dataset"},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )
    max_target_length: Optional[int] = field(
        default=80,
        metadata={"help": "Max input length for the target text"},
    )


class DataProcessor:
    def __init__(self, tokenizer, max_source_length=512, max_target_length=80):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        self.hl_token = "<hl>"            
        self.sep_token = "<sep>"

  
    def process(self, dataset):
        dataset = dataset.map(self._add_eos_examples)
        
        dataset = dataset.map(self._add_special_tokens)
        dataset = dataset.map(self._convert_to_features, batched=True)
        
        return dataset
  
    def _add_eos_examples(self, example):
        example['source_text'] = example['source_text'] + " </s>"
        example['target_text'] = example['target_text'] + " </s>"
        return example
  
    def _add_special_tokens(self, example):
        example['source_text'] = example['source_text'].replace("{hl_token}", self.hl_token)    
        example['target_text'] = example['target_text'].replace("{sep_token}", self.sep_token)
        return example
  
    # tokenize the examples
    def _convert_to_features(self, example_batch):
        source_encoding = self.tokenizer.batch_encode_plus(
            example_batch['source_text'],
            max_length=self.max_source_length,
            padding='max_length',
            pad_to_max_length=True,
            truncation=True, 
        )
        target_encoding = self.tokenizer.batch_encode_plus(
            example_batch['target_text'],
            max_length=self.max_target_length,
            padding='max_length',
            pad_to_max_length=True,
            truncation=True, 
        )

        encodings = {
            'source_ids': source_encoding['input_ids'], 
            'target_ids': target_encoding['input_ids'],
            'attention_mask': source_encoding['attention_mask'],
        }

        return encodings
    

def main():
    parser = HfArgumentParser((DataTrainingArguments,))

    data_args = parser.parse_args_into_dataclasses()[0]

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")

    
    tokenizer.add_tokens(['<sep>', '<hl>'])
    
    train_dataset = Dataset.from_pandas(pd.DataFrame(generate_data(mode="train", path=os.path.join(data_args.input_path, data_args.input_train_file))))
    valid_dataset = Dataset.from_pandas(pd.DataFrame(generate_data(mode="valid", path=os.path.join(data_args.input_path, data_args.input_eval_file))))

    processor = DataProcessor(
        tokenizer,
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length
    )

    
    train_dataset = processor.process(train_dataset)
    valid_dataset = processor.process(valid_dataset)

    columns = ["source_ids", "target_ids", "attention_mask"]
    train_dataset.set_format(type='torch', columns=columns)
    valid_dataset.set_format(type='torch', columns=columns)

    if data_args.output_train_file is None or data_args.output_eval_file is None:
        train_file_name = "train_data_multitask_mt5.pt"
        train_path = os.path.join(data_args.output_path, train_file_name)

        valid_file_name = "valid_data_multitask_mt5.pt"
        valid_path = os.path.join(data_args.output_path, valid_file_name)
    else:
        train_path = os.path.join(data_args.output_path, data_args.output_train_file)
        valid_path = os.path.join(data_args.output_path, data_args.output_eval_file)
    
    torch.save(train_dataset, train_path)
    logger.info(f"saved train dataset at {train_path}")
    
    torch.save(valid_dataset, valid_path)
    logger.info(f"saved validation dataset at {valid_path}")
    
    if not os.path.exists(data_args.tokenizer_path):
        os.mkdir(data_args.tokenizer_path)
    tokenizer.save_pretrained(data_args.tokenizer_path)
    logger.info(f"saved tokenizer at {data_args.tokenizer_path}")


if __name__ == "__main__":
    main()
