import torch
import torch.nn as nn
import math
from torch import Tensor
from torch.nn import Transformer
import transformers_preprocess as data_preprocess
import numpy as np
from tqdm import tqdm
import pickle
from metrics import Metrics,CosineSimilarity,BLEUScore,METEORScore,ROUGEScore_custom
import time

# disable torch warnings
import warnings
warnings.filterwarnings("ignore")


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)

        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)
    
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

        input_ids = np.zeros((n, max_len+2), dtype=np.int32)
        target_ids = np.zeros((n, max_len+2), dtype=np.int32)

        input_ids.fill(data_preprocess.PAD_token)
        target_ids.fill(data_preprocess.PAD_token)

        for idx, (inp, tgt) in tqdm(enumerate(pairs)):
            inp_ids = self.indx_sent_func(input_lang, inp)

            inp_ids.insert(0, self.SOS_token)
            inp_ids.append(self.EOS_token)
            inp_ids = torch.Tensor(inp_ids)

            tgt_ids = self.indx_sent_func(output_lang, tgt)
            tgt_ids.insert(0, self.SOS_token)
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

        train_input_ids = torch.LongTensor(input_ids[:int(0.8*len(input_ids))]).to(self.device)
        train_target_ids = torch.LongTensor(target_ids[:int(0.8*len(target_ids))]).to(self.device)

        test_input_ids = torch.LongTensor(input_ids[int(0.8*len(input_ids)):]).to(self.device)
        test_target_ids = torch.LongTensor(target_ids[int(0.8*len(target_ids)):]).to(self.device)

        train_pairs = pairs[:int(0.8*len(pairs))]
        test_pairs = pairs[int(0.8*len(pairs)):]

        res_dict = {
            'input_lang':input_lang,
            'output_lang':output_lang,
            'train_input_ids':train_input_ids,
            'train_target_ids':train_target_ids,
            'train_pairs':train_pairs,
            'test_input_ids':test_input_ids,
            'test_target_ids':test_target_ids,
            'test_pairs':test_pairs,
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

        save_str = f"langs/{in_lang_name}_to_{out_lang_name}_lang.pkl"

        if not self.use_pkl_data:
            if self.from_lang_str == 'cmn':
                loads = LangTensor(self.device,self.convert_eng_to_lang,True).get_dataloader(self.from_lang_str,self.max_length,self.batch_size)
            else:   
                loads = LangTensor(self.device,self.convert_eng_to_lang).get_dataloader(self.from_lang_str,self.max_length,self.batch_size)

            save_str = f"langs/{loads['input_lang'].name}_to_{loads['output_lang'].name}_lang.pkl"

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
        self.SOS_token = loads['SOS_token']
        self.EOS_token = loads['EOS_token']
        self.attention_flag = config_dict['use_attention']

        self.metrics = [CosineSimilarity(1),BLEUScore(1),METEORScore(1),ROUGEScore_custom(1)]

        SRC_VOCAB_SIZE = self.in_lang.n_words
        TGT_VOCAB_SIZE = self.out_lang.n_words
        EMB_SIZE = 512
        NHEAD = 8
        FFN_HID_DIM = 512
        NUM_ENCODER_LAYERS = 3
        NUM_DECODER_LAYERS = 3

        self.transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM).to(device)
        
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=data_preprocess.PAD_token)

        self.optimizer = torch.optim.Adam(self.transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    def generate_square_subsequent_mask(self,sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def create_mask(self,src, tgt):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len),device=self.device).type(torch.bool)

        src_padding_mask = (src == data_preprocess.PAD_token).transpose(0, 1)
        tgt_padding_mask = (tgt == data_preprocess.PAD_token).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
    
    def greedy_decode(self, model, fsrc, fsrc_masks,fends, start_symbol):
        outputs = []
        for i in range(len(fsrc)):
            src = fsrc[i].unsqueeze(1).to(self.device)
            src_mask = fsrc_masks[i].to(self.device)
            memory = model.encode(src, src_mask)
            ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.device)
            for _ in range(fends[i][0]-1):
                memory = memory.to(self.device)
                tgt_mask = (self.generate_square_subsequent_mask(ys.size(0))
                            .type(torch.bool)).to(self.device)
                out = model.decode(ys, memory, tgt_mask)
                out = out.transpose(0, 1)
                prob = model.generator(out[:, -1])
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.item()
                ys = torch.cat([ys,
                                torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
                if next_word == data_preprocess.EOS_token:
                    break

            outputs.append(ys)

        return outputs

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
                    decoded_words = decoded_words[1:]
                    strings.append(' '.join(decoded_words))

        return strings

    # actual function to translate input sentence into target language
    def translate(self, model: torch.nn.Module, sentence_batch):
        model.eval()
        masks = []
        ends = []
        sentences = []
        for i in range(sentence_batch.shape[0]):
            current_sentence = sentence_batch[i]
            end_val = (current_sentence == data_preprocess.EOS_token).nonzero(as_tuple=True)[0]
            ends.append(end_val+5)
            current_sentence = current_sentence[:end_val.item()]
            sentences.append(current_sentence)
            src_mask = (torch.zeros(current_sentence.shape[0],current_sentence.shape[0])).type(torch.bool)
            masks.append(src_mask)

        tgt_tokens = self.greedy_decode(
            model,  sentences, masks,ends, start_symbol=data_preprocess.SOS_token)


        strings = self.string_from_X(tgt_tokens,type=1)
        model.train()
        return strings

        # for i in tgt_tokens:
        #     tens = i.flatten().tolist()
        #     string_repr = self.string_from_X(tens,type=1)
        #     print(string_repr)


    def train_epoch(self,in_i,tgt_i,eval_flag=False):
        self.transformer.train()


        total_loss = 0
        total_cos = 0
        total_bleu = 0
        total_meteor = 0
        total_rouge = 0

        input_sentence_batch = None
        target_sentence_batch = None
        valid_str = None

        epoch_time = 0

        for k in range(8):
            choices = np.random.choice(len(in_i), self.batch_size, replace=False)
            time1 = time.time()
            input_tensor = in_i[choices].transpose(0,1)
            target_tensor = tgt_i[choices].transpose(0,1)

            tgt_input = target_tensor[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(input_tensor, tgt_input)

            logits = self.transformer(input_tensor, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

            input_sentence_batch = in_i[choices]
            target_sentence_batch = tgt_i[choices]
            valid_str = input_sentence_batch[:,1:]

            time2 = time.time()

            if eval_flag:
                input_strings = self.string_from_X(input_sentence_batch[:,1:],is_input=True)
                translated_strings = self.translate(self.transformer,valid_str)
                target_strings = self.string_from_X(target_sentence_batch,type=1)
                score_vec = self.eval_while_train(target_strings,translated_strings)

                if k==7:
                    print("Input: {}".format(input_strings[0]))
                    print("Translated: {}".format(translated_strings[0]))
                    print("Target: {}".format(target_strings[0]))
                
            else:
                score_vec = np.zeros(3*len(self.metrics))-1

            time3 = time.time()

            self.optimizer.zero_grad()

            tgt_out = target_tensor[1:, :]

            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()
            total_cos += score_vec[0]
            total_bleu += score_vec[3]
            total_meteor += score_vec[6]
            total_rouge += score_vec[9]

            time4 = time.time()

            epoch_time += (time4-time3)+(time2-time1)

            


        return (total_loss/8,total_cos/8,total_bleu/8,total_meteor/8,total_rouge/8, epoch_time/8)
    
    def eval_while_train(self,gts,mts):
        

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

    
    def train(self):
        last_loss = 9999
        last_cosine = 0
        last_bleu = 0
        last_meteor = 0
        last_rouge = 0
        epoch_list = []
        loss_list = []
        cosine_list = []
        bleu_list = []
        meteor_list = []
        rouge_list = []
        time_taken = 0
        current_saved = 0

        start_time = time.time()
        eval_flag = False
        with open('log.txt','a') as f:
            try:
                for i in tqdm(range(self.epoch_count)):
                    if i%20==0:
                        eval_flag = True
                    else:
                        eval_flag = False
                    train_start = time.time()
                    losses = self.train_epoch(self.in_ids,self.tgt_ids,eval_flag)
                    curr_time = time.time()
                    time_taken += losses[5]

                    if eval_flag:
                        epoch_list.append(i)
                        loss_list.append(losses[0])
                        cosine_list.append(losses[1])
                        bleu_list.append(losses[2])
                        meteor_list.append(losses[3])
                        rouge_list.append(losses[4])
                    else:
                        training_time = curr_time-train_start
                        time_taken += training_time
                        

                    f_str = f"Epoch: {i} Loss: {losses[0]} Cosine: {losses[1]} BLEU: {losses[2]} METEOR: {losses[3]} ROUGE: {losses[4]}\n"

                    if losses[0] < last_loss and losses[1]>last_cosine and losses[2]>last_bleu and losses[3]>last_meteor and losses[4]>last_rouge:
                        print(f_str)
                        torch.save(self.transformer.state_dict(), f"{self.in_lang.name}_to_{self.out_lang.name}_tf.pth")
                        last_loss = losses[0]
                        last_cosine = losses[1]
                        last_bleu = losses[2]
                        last_meteor = losses[3]
                        last_rouge = losses[4]

                    if losses[1]>0.9 and losses[2]>0.75 and losses[3]>0.75 and losses[4]>0.75:
                        f.write(f_str)
                        f.write("Time taken: {}\n".format(time_taken))
                        break

                    if (curr_time-start_time) > 3600*4:
                        f.write(f_str)
                        f.write("Time taken: {}\n".format(time_taken))
                        break

            except Exception as e:
                f.write(f_str)
                f.write("Time taken: {}\n".format(time_taken))
                pass

            plot_dict = {
                'src_lang': self.in_lang.name,
                'tgt_lang': self.out_lang.name,
                'attention':self.attention_flag,
                'rnn_type':self.config_dict['rnn_type'],
                'time':time_taken,
                'epoch':epoch_list,
                'loss':loss_list,
                'cos':cosine_list,
                'bleu':bleu_list,
                'meteor':meteor_list,
                'rouge':rouge_list
                }

            with open(f"plots/{self.in_lang.name}_to_{self.out_lang.name}_data.pkl",'wb') as f:
                pickle.dump(plot_dict,f)

            

            




# config_dict = {
#     "convert_eng_to_lang": False,
#     "epoch_count": 20000,
#     "learning_rate": 0.001,
#     "max_sentence_length": 25,
#     "hidden_size": 1024,
#     "batch_size": 128,
#     "from_lang": "cmn",
#     "use_pkl_data": False,
#     "rnn_type": "gru",
#     "loss_type": "nll",
#     "optimizer": "adam",
#     "use_attention": True
# }


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# trainer = Trainer(config_dict,device=device)
# trainer.train()