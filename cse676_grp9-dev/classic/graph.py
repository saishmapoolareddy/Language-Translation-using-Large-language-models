import matplotlib.pyplot as plt
import numpy as np
import sys
import pickle
import os

pickle_file = sys.argv[1]
with open(pickle_file,'rb') as f:
    plot_dict = pickle.load(f)

src_lang = plot_dict['src_lang']
tgt_lang = plot_dict['tgt_lang']
epoch_list = plot_dict['epoch']
loss_list = plot_dict['loss']
cos_list = plot_dict['cos']
bleu_list = plot_dict['bleu']
meteor_list = plot_dict['meteor']
rouge_list = plot_dict['rouge']

time = plot_dict['time']

time_hh = int(time//3600)
time_mm = int((time%3600)//60)
time_ss = int((time%3600)%60)

plt.plot(epoch_list,loss_list)
plt.plot(epoch_list,cos_list)
plt.plot(epoch_list,bleu_list)
plt.plot(epoch_list,meteor_list)
plt.plot(epoch_list,rouge_list)

plt.legend(['loss','cosine','BLEU','METEOR','ROUGE'])
plt.title(f'{src_lang} to {tgt_lang} Attn:{plot_dict["attention"]} RNN:{plot_dict["rnn_type"]}\nTime taken: {time_hh}h {time_mm}m {time_ss}s')
plt.savefig(f'plots/{src_lang}_{tgt_lang}_{plot_dict["attention"]}_{plot_dict["rnn_type"]}.png')
plt.show()