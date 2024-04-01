import train
import torch
import json

with open('config.json','r') as f:
    base_dict = json.load(f)



# langs = ['fra']
# attns = [True]
# rnns = ['gru']

langs = ['ita','spa','deu','cmn','fra']
attns = [True,False]
rnns = ['gru','lstm']


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for r in rnns:
    for a in attns:
        for l in langs:
            try:
                torch.cuda.empty_cache()
                if r == 'lstm' and a == True:
                    continue
                curr_dict = base_dict.copy()
                curr_dict['from_lang'] = l
                curr_dict['use_attention'] = a
                curr_dict['rnn_type'] = r

                trainer = train.Trainer(curr_dict ,device=device)
                trainer.run()
            except Exception as e:
                print(e)
                continue