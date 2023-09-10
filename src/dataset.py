import config

import torch

class EntityDataset:
    def __init__(self,texts,pos,tags):
        self.texts = texts
        self.pos = pos
        self.tags = tags

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self,item):
        text = self.texts[item]
        pos = self.pos[item]
        tags = self.tags[item]

        ids = []
        target_pos = []
        target_tags = []

        for i,s in enumerate(text):
            inputs = config.TOKENIZER.encode(
                s,
                add_special_tokens = False
            )

            input_len = len(inputs)
            ids.extend(inputs)
            target_pos.extend([pos[i]]*input_len)
            target_tags.extend([tags[i]]*input_len)

            ids = ids[:config.MAX_LEN - 2]
            target_pos = target_pos[:config.MAX_LEN - 2]
            target_tags = target_tags[:config.MAX_LEN - 2]

            ids = [101] + ids + [102]
            target_pos = [0] + target_pos + [0]
            target_tags = [0] + target_tags + [0]

            mask = [1] * len(ids)
            token_type_ids = [0] * len(ids)

            padding_len = config.MAX_LEN - len(ids)

            ids = ids + ([0] * padding_len)
            mask = mask + ([0] * padding_len)
            token_type_ids = token_type_ids + ([0] * padding_len)
            target_tags = target_tags + ([0] * padding_len)
            target_pos = target_pos + ([0] * padding_len)

        return {
            "ids":torch.tensor(ids,dtype = torch.long),
            "mask":torch.tensor(mask,dtype = torch.long),
            "token_type_ids":torch.tensor(token_type_ids,dtype = torch.long),
            "target_tags":torch.tensor(target_tags,dtype = torch.long),
            "target_pos":torch.tensor(target_pos,dtype = torch.long)
            }