import tf_custom as train
import torch
import json

with open('config.json','r') as f:
    base_dict = json.load(f)

langs = ['ita','spa','deu','cmn','fra']


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for l in langs:
    try:
        torch.cuda.empty_cache()
        curr_dict = base_dict.copy()
        curr_dict['from_lang'] = l
        trainer = train.Trainer(curr_dict ,device=device)
        trainer.train()
    except Exception as e:
        print(e)
        continue