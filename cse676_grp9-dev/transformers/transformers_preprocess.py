import unicodedata
import re

UNK_token = 0
PAD_token = 1
SOS_token = 2
EOS_token = 3


class Lang:
    def __init__(self, name, by_char=False):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "UNK", 1: "PAD", 2: "SOS", 3: "EOS"}
        self.n_words = 4 # Count SOS and EOS
        self.by_char = by_char

    def addSentence(self, sentence):
        if self.by_char:
            for word in sentence:
                self.addWord(word)
        else:
            for word in sentence.split(' '):
                self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    # s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()


def readLangs(lang1, eng2lang=True, by_char=False):
    # print("Reading lines...")

    # Read the file and split into lines
    lines = open('../data/%s.txt' % (lang1), encoding='utf-8').\
        read().strip().split('\n')
      
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    pairs = [[p[0],p[1]] for p in pairs]

    # Reverse pairs, make Lang instances
    if not eng2lang:

        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang1, by_char=by_char)
        output_lang = Lang('eng')
    else:
        input_lang = Lang('eng')
        output_lang = Lang(lang1, by_char=by_char)

    return input_lang, output_lang, pairs


def filterPair(in_lang,out_lang,p,max_length):
    # return len(p[0].split(' ')) < MAX_LENGTH and \
    #     len(p[1].split(' ')) < MAX_LENGTH and \
    #     p[1].startswith(eng_prefixes)
    if in_lang.by_char and out_lang.by_char:
        return len(p[0]) < max_length and len(p[1]) < max_length
    elif in_lang.by_char and not out_lang.by_char:
        return len(p[0]) < max_length and len(p[1].split(' ')) < max_length
    elif not in_lang.by_char and out_lang.by_char:
        return len(p[0].split(' ')) < max_length and len(p[1]) < max_length
    else:
        return len(p[0].split(' ')) < max_length and len(p[1].split(' ')) < max_length
    
    # return len(p[0].split(' ')) < MAX_LENGTH and \
    #     len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(in_lang,out_lang,fpairs,max_len):
    return [pair for pair in fpairs if filterPair(in_lang,out_lang,pair,max_len)]

def prepareData(lang1, max_length, reverse=False, by_char=False):
    input_lang, output_lang, pairs = readLangs(lang1, reverse, by_char=by_char)
    # print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(input_lang,output_lang,pairs,max_length)
    # print("Trimmed to %s sentence pairs" % len(pairs))
    # print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    # print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def indexesFromSentence(lang, sentence):
    if lang.by_char:
        return [lang.word2index[word] for word in sentence]
    else:
        return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence,token):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(token)
    return indexes


def tensorsFromPair(in_lang,out_lang,pair):
    input_idxs = tensorFromSentence(in_lang, pair[0])
    target_idxs = tensorFromSentence(out_lang, pair[1])
    return (input_idxs, target_idxs)
