import torch
import models
import data_preprocess
import random
import pickle
import json
from train import LangTensor
from metrics import Metrics,CosineSimilarity,BLEUScore,ROUGEScore_custom,METEORScore
from random import choice
class Evaluator():
    def __init__(self,config_dict,device) -> None:
        self.config_dict = config_dict
        self.hidden_size = config_dict['hidden_size']
        self.batch_size = config_dict['batch_size']
        self.from_lang_str = config_dict['from_lang']
        self.max_length = config_dict['max_sentence_length']
        self.attention_flag = config_dict["use_attention"]

        self.use_pkl_data = config_dict['use_pkl_data']

        self.indx_sent_func = data_preprocess.indexesFromSentence
        self.optimizer = None

        self.device = device

        self.epoch_count = config_dict['epoch_count']
        self.learning_rate = config_dict['learning_rate']

    
        with open('langs/{}_to_{}.pkl'.format(from_lang,to_lang), 'rb') as f:
            loads = pickle.load(f)

        
        self.in_lang = loads['input_lang']
        self.out_lang = loads['output_lang']
        self.in_ids = loads['test_input_ids']
        self.tgt_ids = loads['test_target_ids']
        self.pairs = loads['test_pairs']
        self.SOS_token = loads['SOS_token']
        self.EOS_token = loads['EOS_token']

        print("Evaluating translation from {} to {}".format(self.in_lang.name,self.out_lang.name))

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

        self.attention_flag = True if attn == 'True' else False

        if self.attention_flag:
            self.encoder = models.CustomEncoderRNN(encoder_dict).to(self.device)
            self.decoder = models.CustomAttnDecoderRNN(decoder_dict).to(self.device)
        else:
            self.encoder = models.CustomEncoderRNN(encoder_dict).to(self.device)
            self.decoder = models.CustomDecoderRNN(decoder_dict).to(self.device)

        # base_str = f"{self.from_lang_str}_to_{to_lang}_attn_{self.attention_flag}_rnn_{config_dict['rnn_type']}"
        base_str = f"{from_lang}_to_{to_lang}_attn_{attn}_rnn_{rnn}"

        self.encoder_weights = torch.load(f'models/{base_str}_encoder_best.pth')
        self.decoder_weights = torch.load(f'models/{base_str}_decoder_best.pth')

        self.encoder.load_state_dict(self.encoder_weights)
        self.decoder.load_state_dict(self.decoder_weights)

        self.encoder.eval()
        self.decoder.eval()

        self.tfs = data_preprocess.tensorFromSentence

        

    def evaluate(self,encoder, decoder, sentence, input_lang, output_lang):

        with torch.no_grad():
            input_tensor = torch.IntTensor(self.tfs(input_lang, sentence,self.EOS_token)).to(self.device).view(1, -1)

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden,None,self.SOS_token)
            _, topi = decoder_outputs.topk(1)
            
            decoded_ids = topi.squeeze()

            decoded_words = []
            for idx in decoded_ids:
                if idx.item() == self.EOS_token:
                    break
                decoded_words.append(output_lang.index2word[idx.item()])
        return decoded_words, decoder_attn

    def evaluateRandomly(self,encoder, decoder, n=20):
        with open ('eval.txt','w') as f:
            for i in range(n):
                pair = random.choice(self.pairs)
                print('Input >', pair[0])
                print('Expected =', pair[1])
                output_words, _ = self.evaluate(encoder, decoder, pair[0], self.in_lang, self.out_lang)
                output_sentence = ' '.join(output_words)
                print('Got <', output_sentence)
                os = [output_sentence]
                ref = [pair[1]]
                cs = CosineSimilarity(1).evaluate(ref,os)
                bs = BLEUScore(1).evaluate(ref,os)
                ms = METEORScore(1).evaluate(ref,os)
                rs = ROUGEScore_custom(1).evaluate(ref,os)
                cosine_score = cs[0]
                bleu_score = bs[0]
                meteor_score = ms[0]
                rouge_score = rs[0]
                f.write(f'{pair[0]}\t{pair[1]}\t{output_sentence}\n')
                print('')

    def run(self):
        self.evaluateRandomly(self.encoder, self.decoder)
        

    def web_eval(self,phrase=None,random=None):
        if random:
            pair = choice(self.pairs)
            phrase = pair[0]
            expected = pair[1]
            output_words, _ = self.evaluate(self.encoder, self.decoder, phrase, self.in_lang, self.out_lang)
            output_sentence = ' '.join(output_words)

            # print('Input >', phrase)
            # print('Expected =', expected)
            # print('Got <', output_sentence)

            return phrase,output_sentence
        

        try:
            output_words, _ = self.evaluate(self.encoder, self.decoder, phrase, self.in_lang, self.out_lang)
            output_sentence = ' '.join(output_words)
        except:
            output_sentence = "<NAN these words are not present in my vocabulary>"

        return phrase,output_sentence


    
device = torch.device("cpu")
import sys
with open('config.json','r') as f:
    config = json.load(f)


from_lang = sys.argv[1]
to_lang = sys.argv[2]
attn = sys.argv[3]
rnn = sys.argv[4]

evaluator = Evaluator(config,device=device)
# evaluator.web_eval(random=True)
evaluator.run()
