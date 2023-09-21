import pandas as pd

def preprocess_ner_dataset():
    df = pd.read_csv("input/ner_dataset.csv",encoding="unicode_escape")

    df.ffill(inplace=True)

    df['sentence'] = df.groupby("Sentence #")['Word'].transform(lambda x: " ".join(x))
    df['tags'] = df.groupby("Sentence #")['Tag'].transform(lambda x: ",".join(x))

    df = df[['sentence','tags']].drop_duplicates().reset_index(drop=True)

    df.to_csv("input/dataset_processed.csv")
    return True