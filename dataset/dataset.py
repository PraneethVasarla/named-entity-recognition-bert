import os,sys

sys.path.append(os.getcwd())

from datasets import DatasetDict
import pandas as pd

def keep_first_row(group):
    return group.iloc[0]

df = pd.read_csv("dataset/ner_dataset.csv",encoding="unicode_escape")

df['Sentence #'].ffill(inplace=True)

# df['sentence'] = df.groupby(['Sentence #'])['Word'].transform(lambda x: ' '.join(x))
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

df.to_csv("processed.csv")

print(df.head(20))

class NERDataset:
    def __init__(self,dataset_path=None) -> None:
        if dataset_path is not None:
            df = pd.read_csv("dataset/ner_dataset.csv",encoding="unicode_escape")
            self.df = df
        else:
            raise Exception("Pass dataset path")
        
    def prepare_hf_dataset(self):
        df['Sentence #'].fillna(method='ffill',inplace=True)
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

        return df
