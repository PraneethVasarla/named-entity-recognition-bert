import os,sys

sys.path.append(os.getcwd())

from datasets import Dataset,DatasetDict
import pandas as pd
from sklearn.model_selection import train_test_split

def keep_first_row(group):
    return group.iloc[0]

class NERDataset:
    def __init__(self,dataset_path=None) -> None:
        if dataset_path is not None:
            df = pd.read_csv("dataset/ner_dataset.csv",encoding="unicode_escape")
            self.df = df
        else:
            raise Exception("Pass dataset path")
        
    def prepare_hf_dataset(self):
        df = self.df.copy()

        df['Sentence #'].ffill(inplace=True)
        df['sentence'] = df.groupby(['Sentence #'])['Word'].transform(lambda x: ' '.join(str(word) for word in x))
        df['tokens'] = df['sentence'].apply(lambda x: x.split())
        df.drop("sentence",axis=1,inplace=True)

        df['tags'] = df.groupby(['Sentence #'])['Tag'].transform(lambda x: ' '.join(str(tag) for tag in x))
        df['ner_tags'] = df['tags'].apply(lambda x: x.split())
        df.drop("tags",axis=1,inplace=True)

        df.drop(["Word","POS","Tag"],axis=1,inplace=True)
        df['Sentence #'] = df['Sentence #'].apply(lambda x: int(x.split(" ")[-1]))
        df.rename(columns={"Sentence #":"id"},inplace=True)

        df = df.groupby('id').apply(keep_first_row)
        df = df.set_index('id').sort_index()
        df = df.reset_index()

        # Split the data into train and test sets
        train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)

        # Split the train set into train and validation sets
        train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)

        train_dataset = Dataset.from_pandas(train_df,preserve_index=False)
        val_dataset = Dataset.from_pandas(val_df,preserve_index=False)
        test_dataset = Dataset.from_pandas(test_df,preserve_index=False)

        dataset = DatasetDict()
        dataset['train'] = train_dataset
        dataset['validation'] = val_dataset
        dataset['test'] = test_dataset

        self.hf_dataset = dataset

        return self.hf_dataset
