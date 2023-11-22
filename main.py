from dataset import NERDataset
from src import CustomNERModel

dataset = NERDataset(dataset_path="dataset/ner_dataset.csv")

model_checkpoint = "bert-base-uncased"

model_trainer = CustomNERModel(dataset=dataset,model_checkpoint=model_checkpoint)

model_trainer.train()
