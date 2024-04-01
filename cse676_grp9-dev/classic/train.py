from tqdm import tqdm
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import data_preprocess
import torch
import torch.nn as nn
import torch.optim as optim
import models
import math
import pickle
from metrics import Metrics,CosineSimilarity,BLEUScore,METEORScore,ROUGEScore_custom
import time
#from torch.utils.tensorboard import SummaryWriter

class LangDataLoader():
    def __init__(self,arg_device=None) -> None:
        self.indx_sent_func = data_preprocess.indexesFromSentence
        self.SOS_token = data_preprocess.SOS_token
        self.EOS_token = data_preprocess.SOS_token

        if arg_device is None:
            self.device = torch.device("cpu")

        else:
            self.device = arg_device


    def get_dataloader(self,lang_str='fra',max_len=10,batch_size=32):
        input_lang, output_lang, pairs = data_preprocess.prepareData(lang_str, True)

        n = len(pairs)

        input_ids = np.zeros((n, max_len), dtype=np.int32)
        target_ids = np.zeros((n, max_len), dtype=np.int32)

        for idx, (inp, tgt) in tqdm(enumerate(pairs)):
            inp_ids = self.indx_sent_func(input_lang, inp)
            inp_ids.append(self.EOS_token)
            inp_ids = torch.Tensor(inp_ids)


            tgt_ids = self.indx_sent_func(output_lang, tgt)
            tgt_ids.append(self.EOS_token)
            tgt_ids = torch.Tensor(tgt_ids)

            input_ids[idx, :len(inp_ids)] = inp_ids
            target_ids[idx, :len(tgt_ids)] = tgt_ids

        train_data = TensorDataset(torch.LongTensor(input_ids).to(self.device),
                                torch.LongTensor(target_ids).to(self.device))

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        return input_lang, output_lang, train_dataloader, pairs, self.SOS_token, self.EOS_token
    
class LangTensor():
    def __init__(self,device,eng2lang,strip_chars=False) -> None:
        self.indx_sent_func = data_preprocess.indexesFromSentence
        self.SOS_token = data_preprocess.SOS_token
        self.EOS_token = data_preprocess.EOS_token
        self.device = device
        self.eng2lang = eng2lang
        self.strip_chars = strip_chars


    def get_dataloader(self,lang_str,max_len,batch_size):
        input_lang, output_lang, pairs = data_preprocess.prepareData(lang_str,max_len, self.eng2lang,self.strip_chars)
        n = len(pairs)

        input_ids = np.zeros((n, max_len), dtype=np.int32)
        target_ids = np.zeros((n, max_len), dtype=np.int32)

        train_test_val_split = [0.8,0.1,0.1]

        train_idxs = [0,int(train_test_val_split[0]*n)]
        test_idxs = [int(train_test_val_split[0]*n),int(train_test_val_split[0]*n)+int(train_test_val_split[1]*n)]
        val_idxs = [int(train_test_val_split[0]*n)+int(train_test_val_split[1]*n),n]


        for idx, (inp, tgt) in tqdm(enumerate(pairs)):
            inp_ids = self.indx_sent_func(input_lang, inp)

            inp_ids.append(self.EOS_token)
            inp_ids = torch.Tensor(inp_ids)


            tgt_ids = self.indx_sent_func(output_lang, tgt)
            tgt_ids.append(self.EOS_token)
            tgt_ids = torch.Tensor(tgt_ids)
            input_ids[idx, :len(inp_ids)] = inp_ids
            target_ids[idx, :len(tgt_ids)] = tgt_ids

        print("Generating train and test data")
        # shuffle
        print("Length of input_ids: {}".format(len(input_ids)))
        rand_idxs = np.random.permutation(len(input_ids))
        input_ids = input_ids[rand_idxs]
        target_ids = target_ids[rand_idxs]

        pairs = np.array(pairs)[rand_idxs]

        # train_input_ids = torch.LongTensor(input_ids[:int(0.8*len(input_ids))]).to(self.device)
        # train_target_ids = torch.LongTensor(target_ids[:int(0.8*len(target_ids))]).to(self.device)

        # test_input_ids = torch.LongTensor(input_ids[int(0.8*len(input_ids)):]).to(self.device)
        # test_target_ids = torch.LongTensor(target_ids[int(0.8*len(target_ids)):]).to(self.device)

        # train_pairs = pairs[:int(0.8*len(pairs))]
        # test_pairs = pairs[int(0.8*len(pairs)):]

        train_input_ids = torch.LongTensor(input_ids[train_idxs[0]:train_idxs[1]]).to(self.device)
        train_target_ids = torch.LongTensor(target_ids[train_idxs[0]:train_idxs[1]]).to(self.device)

        test_input_ids = torch.LongTensor(input_ids[test_idxs[0]:test_idxs[1]]).to(self.device)
        test_target_ids = torch.LongTensor(target_ids[test_idxs[0]:test_idxs[1]]).to(self.device)

        val_input_ids = torch.LongTensor(input_ids[val_idxs[0]:val_idxs[1]]).to(self.device)
        val_target_ids = torch.LongTensor(target_ids[val_idxs[0]:val_idxs[1]]).to(self.device)

        train_pairs = pairs[train_idxs[0]:train_idxs[1]]
        test_pairs = pairs[test_idxs[0]:test_idxs[1]]
        val_pairs = pairs[val_idxs[0]:val_idxs[1]]

        
        res_dict = {
            'input_lang':input_lang,
            'output_lang':output_lang,
            'train_input_ids':train_input_ids,
            'train_target_ids':train_target_ids,
            'train_pairs':train_pairs,
            'test_input_ids':test_input_ids,
            'test_target_ids':test_target_ids,
            'test_pairs':test_pairs,
            'val_input_ids':val_input_ids,
            'val_target_ids':val_target_ids,
            'val_pairs':val_pairs,
            'SOS_token':self.SOS_token,
            'EOS_token':self.EOS_token
        }

        print("Train size: {} Test size: {}".format(len(train_input_ids),len(test_input_ids)))

        return res_dict
    

class Trainer():
    
    def __init__(self,config_dict,device) -> None:

        self.config_dict = config_dict
        self.hidden_size = config_dict['hidden_size']
        self.batch_size = config_dict['batch_size']
        self.from_lang_str = config_dict['from_lang']
        self.max_length = config_dict['max_sentence_length']
        self.convert_eng_to_lang = config_dict['convert_eng_to_lang']

        self.use_pkl_data = config_dict['use_pkl_data']

        self.indx_sent_func = data_preprocess.indexesFromSentence
        self.optimizer = None

        self.device = device

        self.epoch_count = config_dict['epoch_count']
        self.learning_rate = config_dict['learning_rate']

        import pprint
        pprint.pprint(config_dict)

        in_lang_name = "eng" if self.convert_eng_to_lang else self.from_lang_str
        out_lang_name = self.from_lang_str if self.convert_eng_to_lang else "eng"

        save_str = f"langs/{in_lang_name}_to_{out_lang_name}.pkl"

        if not self.use_pkl_data:
            if self.from_lang_str == 'cmn':
                loads = LangTensor(self.device,self.convert_eng_to_lang,True).get_dataloader(self.from_lang_str,self.max_length,self.batch_size)
            else:   
                loads = LangTensor(self.device,self.convert_eng_to_lang).get_dataloader(self.from_lang_str,self.max_length,self.batch_size)

            save_str = f"langs/{loads['input_lang'].name}_to_{loads['output_lang'].name}.pkl"

            with open(save_str, 'wb') as f:
                pickle.dump(loads, f)
        
        else:
            with open(save_str, 'rb') as f:
                loads = pickle.load(f)


        self.in_lang = loads['input_lang']
        self.out_lang = loads['output_lang']
        self.in_ids = loads['train_input_ids']
        self.tgt_ids = loads['train_target_ids']
        self.pairs = loads['train_pairs']
        self.val_in_ids = loads['val_input_ids']
        self.val_tgt_ids = loads['val_target_ids']
        self.val_pairs = loads['val_pairs']
        self.SOS_token = loads['SOS_token']
        self.EOS_token = loads['EOS_token']
        self.attention_flag = config_dict['use_attention']


        # print("Training for translating from {} to {}".format(self.from_lang_str,self.out_lang.name))

        self.metrics = [CosineSimilarity(self.batch_size),BLEUScore(self.batch_size),METEORScore(self.batch_size),ROUGEScore_custom(self.batch_size)]
        # self.metrics = [ROUGEScore_custom(self.batch_size)]

        # self.encoder = models.EncoderRNN(self.in_lang.n_words, self.hidden_size).to(self.device)
        # self.decoder = models.DecoderRNN(self.hidden_size, self.out_lang.n_words, self.max_length,device).to(self.device)
        
        encoder_dict = {
            'input_size':self.in_lang.n_words,
            'hidden_size':self.hidden_size,
            'recurrent_arch': config_dict['rnn_type'],
            'dropout_p':0.1
        }

        decoder_dict = {
            'hidden_size':self.hidden_size,
            'output_size':self.out_lang.n_words,
            'recurrent_arch':config_dict['rnn_type'],
            'max_len':self.max_length,
            'arg_device':self.device
        }

        if self.attention_flag:
            self.encoder = models.CustomEncoderRNN(encoder_dict).to(self.device)
            self.decoder = models.CustomAttnDecoderRNN(decoder_dict).to(self.device)

        else:
            self.encoder = models.CustomEncoderRNN(encoder_dict).to(self.device)
            self.decoder = models.CustomDecoderRNN(decoder_dict).to(self.device)

        if config_dict['optimizer'] == 'sgd':
            self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=self.learning_rate)
            self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=self.learning_rate)
        elif config_dict['optimizer'] == 'rmsprop':
            self.encoder_optimizer = optim.RMSprop(self.encoder.parameters(), lr=self.learning_rate)
            self.decoder_optimizer = optim.RMSprop(self.decoder.parameters(), lr=self.learning_rate)
        else:
            self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
            self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.learning_rate)


    def string_from_X(self,decoder_outputs,type=0,is_input=False):
        strings = []
        if is_input:
            for i in decoder_outputs:
                decoded_ids = i
                decoded_words = []
                for idx in decoded_ids:
                    if idx == self.EOS_token:
                        break
                    decoded_words.append(self.in_lang.index2word[idx.item()])
                strings.append(' '.join(decoded_words))
        else:
            if type == 0:
                for i in decoder_outputs:
                    _, topi = i.topk(1)
                    decoded_ids = topi.squeeze()

                    decoded_words = []
                    for idx in decoded_ids:
                        if idx.item() == self.EOS_token:
                            break
                        decoded_words.append(self.out_lang.index2word[idx.item()])
                    strings.append(' '.join(decoded_words))
            else:
                for i in decoder_outputs:
                    decoded_ids = i
                    decoded_words = []
                    for idx in decoded_ids:
                        if idx == self.EOS_token:
                            break
                        decoded_words.append(self.out_lang.index2word[idx.item()])
                    strings.append(' '.join(decoded_words))

        return strings
    
    def train_epoch(self,in_i,tgt_i, encoder, decoder, encoder_optimizer,
            decoder_optimizer, criterion, eval):

        total_loss = 0
        total_cos = 0
        total_bleu = 0
        total_meteor = 0
        total_rouge = 0
        for k in range(8):
            choices = np.random.choice(len(in_i), self.batch_size, replace=False)
            val_choices = np.random.choice(len(self.val_in_ids), self.batch_size, replace=False)
            # choices = [ 17583, 170967, 125800,   4627, 158706]

            input_tensor = in_i[choices]
            target_tensor = tgt_i[choices]

            

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor,self.SOS_token)

            if eval:
                encoder.eval()
                decoder.eval()

                input_val_tensor = self.val_in_ids[val_choices]
                target_val_tensor = self.val_tgt_ids[val_choices]

                vencoder_outputs, vencoder_hidden = encoder(input_val_tensor)
                vdecoder_outputs, _, _ = decoder(vencoder_outputs, vencoder_hidden, target_val_tensor,self.SOS_token)

                generated = self.string_from_X(vdecoder_outputs,0,False)
                inputs = self.string_from_X(input_val_tensor,1,True)
                targets = self.string_from_X(target_val_tensor,1,False)

                score_vec = self.eval_while_train(targets,generated)

                if k==7:
                    print("Input: {}".format(inputs[0]))
                    print("Target: {}".format(targets[0]))
                    print("Generated: {}".format(generated[0]))

                encoder.train()
                decoder.train()

            else:
                score_vec = np.zeros(3*len(self.metrics))-1

            cos_sim = score_vec[0]
            bleu_score = score_vec[3]
            meteor_score = score_vec[6]
            rouge_score = score_vec[9]
            

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )
            loss.backward()

            total_loss += loss.item()
            total_cos += cos_sim
            total_bleu += bleu_score
            total_meteor += meteor_score
            total_rouge += rouge_score

            encoder_optimizer.step()
            decoder_optimizer.step()

        
        return total_loss/8, [total_cos/8,total_bleu/8,total_meteor/8,total_rouge/8]
        # return loss.item(), cos_sim


    def eval_while_train(self,gts,mts):
        # cos_sim,max_cos,min_cos = self.metrics.evaluate(gts,mts)

        m = Metrics()

        cleaned_gts,cleaned_mts = m.clean(gts,mts)

        score_vector = np.zeros(3*len(self.metrics))
        
        for i in range(len(self.metrics)):
            if i==0:
                score_tup = self.metrics[i].evaluate(cleaned_gts,cleaned_mts)
                score_vector[i*3] = score_tup[0]
                score_vector[i*3+1] = score_tup[1]
                score_vector[i*3+2] = score_tup[2]
            else:
                score_tup = self.metrics[i].evaluate(gts,mts)
                score_vector[i*3] = score_tup[0]
                score_vector[i*3+1] = score_tup[1]
                score_vector[i*3+2] = score_tup[2]

        return score_vector


    def run(self):

        last_loss = math.inf
        last_cos = 0
        last_bleu = 0
        last_meteor = 0
        last_rouge = 0

        criterion = nn.NLLLoss()
        # criterion = nn.CrossEntropyLoss()

        epoch_list = []
        loss_list = []
        cos_list = []
        bleu_list = []
        meteor_list = []
        rouge_list = []

        eval_flag = False
        start = time.time()
    
        total_time = 0
        with open('log.txt','a') as f:
            f.write(f"Lang from:{self.from_lang_str} Lang to:{self.out_lang.name}\n")
            f.write(f"Attention:{self.attention_flag}\n")
            f.write(f"RNN type:{self.config_dict['rnn_type']}\n")
    
        f_str = ""
        model_save_str = f"{self.from_lang_str}_to_{self.out_lang.name}_attn_{self.config_dict['use_attention']}_rnn_{self.config_dict['rnn_type']}"
        try:
            for epoch in tqdm(range(0, self.epoch_count)):

                if epoch % 20 == 0:
                    eval_flag = True
                else:
                    eval_flag = False

                if not eval_flag:
                    train_start = time.time()

                loss,scores = self.train_epoch(self.in_ids,self.tgt_ids, self.encoder, self.decoder, self.encoder_optimizer, self.decoder_optimizer, criterion,eval_flag)
                curr_time = time.time()

                if eval_flag:
                    epoch_list.append(epoch)
                    loss_list.append(loss)
                    cos_list.append(scores[0])
                    bleu_list.append(scores[1])
                    meteor_list.append(scores[2])
                    rouge_list.append(scores[3])
                    f_str = 'Epoch:{} Loss:{:.4f} Cos:{:.6f} BLEU:{:.6f} METEOR:{:.6f} ROUGE:{:.6f}\n'.format(epoch, loss,scores[0],scores[1],scores[2],scores[3])
                    with open('log.txt','a') as f:
                        f.write(f_str)  
                else:
                    training_time = curr_time - train_start
                    total_time += training_time

                f_str = 'Epoch:{} Loss:{:.4f} Cos:{:.6f} BLEU:{:.6f} METEOR:{:.6f} ROUGE:{:.6f}\n'.format(epoch, loss,scores[0],scores[1],scores[2],scores[3])

                if loss < last_loss and scores[0]>last_cos and scores[1]>last_bleu and scores[2]>last_meteor and scores[3]>last_rouge:
                    print(f_str)
                    last_loss = loss
                    last_cos = scores[0]
                    last_bleu = scores[1]
                    last_meteor = scores[2]
                    last_rouge = scores[3]
                    torch.save(self.encoder.state_dict(), 'models/{}_encoder_best.pth'.format(model_save_str))
                    torch.save(self.decoder.state_dict(), 'models/{}_decoder_best.pth'.format(model_save_str))

                if scores[0]>0.9 and scores[1]>0.75 and scores[2]>0.75 and scores[3]>0.75:
                    f_str = 'Epoch:{} Loss:{:.4f} Cos:{:.6f} BLEU:{:.6f} METEOR:{:.6f} ROUGE:{:.6f}\n'.format(epoch, loss,scores[0],scores[1],scores[2],scores[3])
                    print(f_str)
                    last_loss = loss
                    torch.save(self.encoder.state_dict(), 'models/{}_encoder_best.pth'.format(model_save_str))
                    torch.save(self.decoder.state_dict(), 'models/{}_decoder_best.pth'.format(model_save_str))
                    break

                if curr_time-start > 3600*4:
                    break


        except Exception as e:
            print(e)
            pass

        

        total_secs = total_time

        with open('log.txt','a') as f:
            f.write(f"Time taken: {total_secs} mins\n")
            f.write("====================================\n")
        
        plot_dict = {
            'src_lang': self.in_lang.name,
            'tgt_lang': self.out_lang.name,
            'attention':self.attention_flag,
            'rnn_type':self.config_dict['rnn_type'],
            'time':total_secs,
            'epoch':epoch_list,
            'loss':loss_list,
            'cos':cos_list,
            'bleu':bleu_list,
            'meteor':meteor_list,
            'rouge':rouge_list
        }

        import pickle
        with open(f"plots/{model_save_str}.pkl",'wb') as f:
            pickle.dump(plot_dict,f)

        return
