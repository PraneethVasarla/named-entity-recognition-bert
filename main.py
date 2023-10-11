from dataset import NERDataset

dataset = NERDataset(dataset_path="dataset/ner_dataset.csv")

hf = dataset.prepare_hf_dataset()

print(hf)