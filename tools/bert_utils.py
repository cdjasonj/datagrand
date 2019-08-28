import codecs
from keras_bert import Tokenizer
import numpy as np
from tqdm import tqdm
import json

maxlen = 128
#bert_path


dict_path = './Bert/vocab.txt'
#
id2char, char2id = json.load(open('./inputs/char2id.json', encoding='utf-8'))
id2bichar, bichar2id = json.load(open('./inputs/bichar2id.json', encoding='utf-8'))
id2BIO, BIO2id = json.load(open('./inputs/bio2id.json', encoding='utf-8'))

def seq_padding(X):
    # L = [len(x) for x in X]
    # ML =
    return [x + [0] * (maxlen - len(x)) if len(x) < maxlen else x[:maxlen] for x in X]

def encode(text):
    vocabs = set()
    with open(dict_path, encoding='utf8') as f:
        for l in f:
            vocabs.add(l.replace('\n', ''))

    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    tokenizer = Tokenizer(token_dict)
    tokens = ['[CLS]'] + [ch if ch in vocabs else '[UNK]' for ch in text] + ['[SEP]']
    return tokenizer._convert_tokens_to_ids(tokens), [0] * len(tokens)

class data_generator:
    def __init__(self, data,batch_size=32):
        self.data = data
        self.batch_size = batch_size

        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
        self.token_dict = {}
        with codecs.open(dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                self.token_dict[token] = len(self.token_dict)
        self.tokenizer = Tokenizer(self.token_dict)
        self.cache_data = []
        self.vocabs = set()
        with open(dict_path, encoding='utf8') as f:
            for l in f:
                self.vocabs.add(l.replace('\n', ''))

    def init_cache_data(self):
        cur_step = 0
        for i, t in enumerate(self.get_next()):
            if i >= self.steps:
                break
            cur_step += 1
            self.cache_data.append(t)

    def __len__(self):
        return self.steps

    def encode(self, text):
        tokens = ['[CLS]'] + [ch if ch in self.vocabs else '[UNK]' for ch in text] + ['[SEP]']
        return self.tokenizer._convert_tokens_to_ids(tokens), [0] * len(tokens)
    def __iter__(self):
        while True:
            idxs = [i for i in range(len(self.data))]
            np.random.shuffle(idxs)
            BERT_INPUT0, BERT_INPUT1,BICHAR_INPUT,BIO = [],[],[],[]
            for i in idxs:
                _data = self.data[i]
                text = _data['text']
                or_text = text
                indices, segments = self.encode(or_text)

                #前后要加上填充两个无用字符
                _bio = [BIO2id.get(bio,0) for bio in _data['bio']]
                # _bichar = [bichar2id.get(bichar,0) for bichar in _data['bichar']]

                #在前后插入0 作为pad
                # _bichar.insert(0,0)
                # _bichar.append(0)

                _bio.insert(0, 0)
                _bio.append(0)

                BERT_INPUT0.append(indices)
                BERT_INPUT1.append(segments)
                # BICHAR_INPUT.append(_bichar)
                BIO.append(_bio)

                if len(BERT_INPUT1) == self.batch_size or i == idxs[-1]:
                    BERT_INPUT0 = np.array(seq_padding(BERT_INPUT0))
                    BERT_INPUT1 = np.array(seq_padding(BERT_INPUT1))
                    # BICHAR_INPUT = np.array(seq_padding(BICHAR_INPUT))
                    BIO = np.array(seq_padding(BIO))

                    yield [BERT_INPUT0, BERT_INPUT1,BIO], None
                    BERT_INPUT0, BERT_INPUT1,BIO= [],[],[]


class test_data_generator:
    def __init__(self, data,batch_size=32):
        self.data = data
        self.batch_size = batch_size

        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
        self.token_dict = {}
        with codecs.open(dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                self.token_dict[token] = len(self.token_dict)
        self.tokenizer = Tokenizer(self.token_dict)
        self.cache_data = []
        self.vocabs = set()
        with open(dict_path, encoding='utf8') as f:
            for l in f:
                self.vocabs.add(l.replace('\n', ''))

    def init_cache_data(self):
        cur_step = 0
        for i, t in enumerate(self.get_next()):
            if i >= self.steps:
                break
            cur_step += 1
            self.cache_data.append(t)

    def __len__(self):
        return self.steps

    def encode(self, text):
        tokens = ['[CLS]'] + [ch if ch in self.vocabs else '[UNK]' for ch in text] + ['[SEP]']
        return self.tokenizer._convert_tokens_to_ids(tokens), [0] * len(tokens)
    def __iter__(self):
        while True:
            idxs = [i for i in range(len(self.data))]

            BERT_INPUT0, BERT_INPUT1,BICHAR_INNPUT = [],[],[]
            for i in idxs:
                _data = self.data[i]
                text = _data['text']
                or_text = text
                indices, segments = self.encode(or_text)

                #前后要加上填充两个无用字符
                _bio = [BIO2id.get(bio,0) for bio in _data['bio']]
                # _bichar = [bichar2id.get(bichar,0) for bichar in _data['bichar']]
                #在前后插入0 作为pad
                # _bichar.insert(0,0)
                # _bichar.append(0)
                _bio.insert(0, 0)
                _bio.append(0)

                BERT_INPUT0.append(indices)
                BERT_INPUT1.append(segments)
                # BICHAR_INNPUT.append(_bichar)
                if len(BERT_INPUT1) == self.batch_size or i == idxs[-1]:
                    BERT_INPUT0 = np.array(seq_padding(BERT_INPUT0))
                    BERT_INPUT1 = np.array(seq_padding(BERT_INPUT1))
                    # BICHAR_INNPUT = np.array(seq_padding(BICHAR_INNPUT))
                    yield [BERT_INPUT0, BERT_INPUT1], None
                    BERT_INPUT0, BERT_INPUT1= [],[]