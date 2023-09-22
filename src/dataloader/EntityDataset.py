import os,sys
sys.path.append(os.getcwd())

import configparser
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils import get_torch_gpu_device
from input import preprocess_ner_dataset
from transformers import BertTokenizerFast,BertConfig,BertForTokenClassification

device = get_torch_gpu_device()

input_dir_files = os.listdir("input")

if "dataset_processed.csv" not in os.listdir("input"):
    preprocess_ner_dataset()
    print("Data has been processed")
else:
    print("Preprocessed data available")


config = configparser.ConfigParser()
config.read("config/project_config.ini")

MAX_LEN = config['bert']['MAX_LEN']
TRAIN_BATCH_SIZE = config['bert']['TRAIN_BATCH_SIZE']
VALID_BATCH_SIZE = config['bert']['VALID_BATCH_SIZE']
EPOCHS = config['bert']['EPOCHS']
LEARNING_RATE = config['bert']['LEARNING_RATE']
MAX_GRAD_NORM = config['bert']['MAX_GRAD_NORM']

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def load_json(path):
    with open(path,"r") as f:
        data = json.load(f)
    return data

class EntityDataset:
    def __init__(self,dataframe,tokenizer,max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.ids_to_labels = load_json("input/ids_to_labels.json")
        self.labels_to_ids = load_json("input/labels_to_ids.json")

    def __getitem__(self,index):
        sentence = self.data.sentence[index].strip()
        tags = self.data.tags[index].split(",")

        encoded = self.tokenizer(sentence,
                                # is_pretokenized=True,
                                old_offset_mapping=True,
                                padding='max_length',
                                truncation=True,
                                max_length=self.max_len)
        
        labels = [self.labels_to_ids[label] for label in tags]
        encoded_labels = np.ones(len(encoded['offset_mapping']),dtype=int) * -100

        i = 0
        for idx,mapping in enumerate(encoded['offset_mapping']):
            if mapping[0] == 0 and mapping[1]!= 0:
                encoded_labels[idx] = labels[i]
                i+=1

        item = {key: torch.as_tensor(val) for key, val in encoded.items()}
        item["labels"] = torch.as_tensor(encoded_labels)

        return item
    
    def __len__(self):
        return self.len
