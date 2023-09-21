import os,sys
sys.path.append(os.getcwd())

import configparser
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

print(config["bert"]['MAX_LEN'])
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

