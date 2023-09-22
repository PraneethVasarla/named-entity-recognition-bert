import pandas as pd
import json

def preprocess_ner_dataset():
    df = pd.read_csv("input/ner_dataset.csv",encoding="unicode_escape")

    df.ffill(inplace=True)

    df['sentence'] = df.groupby("Sentence #")['Word'].transform(lambda x: " ".join(x))
    df['tags'] = df.groupby("Sentence #")['Tag'].transform(lambda x: ",".join(x))

    labels_to_ids = {k: v for v, k in enumerate(df.Tag.unique())}
    ids_to_labels = {v: k for v, k in enumerate(df.Tag.unique())}

    with open("input/labels_to_ids.json","w") as file:
        json.dump(labels_to_ids,file)
    
    with open("input/ids_to_labels.json","w") as file:
        json.dump(ids_to_labels,file)

    df = df[['sentence','tags']].drop_duplicates().reset_index(drop=True)
    df.to_csv("input/dataset_processed.csv")

    return True