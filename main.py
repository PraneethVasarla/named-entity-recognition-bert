from transformers import BertTokenizerFast
import pandas as pd
from src import EntityDataset

df = pd.read_csv("input/dataset_processed.csv")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

train = EntityDataset(df,tokenizer=tokenizer,max_len=128)

print(train[0])