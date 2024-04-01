import torch
import models
import data_preprocess
import random
import pickle
import json
import io   #myadd1
import os #myadd2
import sys  #myadd3
from metrics import CosineSimilarity,BLEUScore,ROUGEScore_custom,METEORScore
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

    
        #with open('{}_to_{}.pkl'.format(from_lang,to_lang), 'rb') as f:
        with open('C:\\Users\\amrut\\Downloads\\eval\\eval\\fra_to_eng.pkl','rb') as f:
    
        #     loads = pickle.load(f)
        
        #with open('{}_to_{}.pkl'.format(from_lang,to_lang), 'rb') as f:
             loads = CPU_Unpickler(f).load()

        
        self.in_lang = loads['input_lang'] #Input language model
        self.out_lang = loads['output_lang'] #Output language model
        self.in_ids = loads['test_input_ids'] #Input language as integer vectors
        self.tgt_ids = loads['test_target_ids'] #Target language as integer vectors
        self.pairs = loads['test_pairs'] #String pairs of input and target language
        self.SOS_token = loads['SOS_token']
        self.EOS_token = loads['EOS_token']

        print("Evaluating translation from {} to {}".format(self.in_lang.name,self.out_lang.name))
        
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
        
        base_dir = 'C:\\Users\\amrut\\Downloads\\eval\\eval'

        base_str = f"{from_lang}_to_{to_lang}_attn_{attn}_rnn_{rnn}"

        # self.encoder_weights = torch.load(f'{base_str}_encoder_best.pth')
        # self.decoder_weights = torch.load(f'{base_str}_decoder_best.pth')
        self.encoder_weights = torch.load(os.path.join(base_dir, f'{base_str}_encoder_best.pth'), map_location=torch.device('cpu'))
        self.decoder_weights = torch.load(os.path.join(base_dir, f'{base_str}_decoder_best.pth'), map_location=torch.device('cpu'))

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

    def evaluateRandomly(self,encoder, decoder, n=1):
        for i in range(num_evals):
            pair = random.choice(self.pairs)
            print('Input >', pair[0])
            print('Expected =', pair[1])
            output_words, _ = self.evaluate(encoder, decoder, pair[0], self.in_lang, self.out_lang)
            output_sentence = ' '.join(output_words)
            print('Got <', output_sentence)
            print('\n')
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

    def run(self):
        self.evaluateRandomly(self.encoder, self.decoder)
        

    def web_eval(self,input_phrase=None,randomarg=False):
        # print("The input phrase entered by the user is: ", input_phrase)
        pair_count = len(self.pairs)

        if randomarg:
            randomidx = random.randint(0,pair_count)
            #pair = choice(self.pairs)
            pair = self.pairs[randomidx]
            
        else:
        # Search for the input_phrase in the first elements of pairs
            try:
                idx = [pair[0] for pair in self.pairs].index(input_phrase)
                pair = self.pairs[idx]
            except ValueError:
                return input_phrase, "<NAN: Input not found in dataset>"    
            phrase = pair[0]
            expected = pair[1]
            output_words, _ = self.evaluate(self.encoder, self.decoder, phrase, self.in_lang, self.out_lang)
            output_sentence = ' '.join(output_words)

            # print('Input >', phrase)
            # print('Expected =', expected)
            # print('Got <', output_sentence)

            #phrase,
            return expected, output_sentence
    
device = torch.device("cpu")
import sys
config_path = r'C:\Users\amrut\Downloads\eval\eval\config.json'
with open(config_path,'r') as f:
    config = json.load(f)

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

from_lang = 'fra'
to_lang = 'eng'
attn = 'True'
rnn = 'gru'
num_evals = 5
#evaluator = Evaluator(config,device=device)
# if __name__ == "__main__":
#     #if len(sys.argv) > 1:
#         #input_phrase = sys.argv[1]
#         evaluator.web_eval(input_phrase =input_phrase, randomarg=False)
#     #else:
#         #print("Please provide an input phrase.")
