from neuralnets.ELMo.elmo_bilstm_cnn_crf import Bilstm_cnn_crf as bilstm_cnn
from neuralnets.ELMo.elmo_local_att_gru import Local_att_gru as local_att_gru
from neuralnets.ELMo.elmo_bilstm_selatt_crf import  Bilstm_selfatt_crf as bilstm_self_att
from tqdm import tqdm
from bilm import TokenBatcher
from tools.load_embedding_matrix import load_embedding
import codecs
import json
import numpy as np
import os
from keras import backend as K
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
train_data = json.load(open('./inputs/train_data.json', encoding='utf-8'))
test_data = json.load(open('./inputs/test_data.json', encoding='utf-8'))
id2char, char2id = json.load(open('./inputs/char2id.json', encoding='utf-8'))
id2bichar, bichar2id = json.load(open('./inputs/bichar2id.json', encoding='utf-8'))
id2BIO, BIO2id = json.load(open('./inputs/bio2id.json', encoding='utf-8'))

vocab_file = './ELMo/DaGuanVocabForElmo.txt'
batcher = TokenBatcher(vocab_file)

params = {'char2id_size':len(char2id),'epochs':100,'early_stopping':8,'bichar2id_size':len(bichar2id),'n_class_labels':len(BIO2id)}

def process_batch_data(batch_data,char2id,BIO2id,mode):

    new_batch_data = []
    elmo_text = []
    if mode == 'dev':
        for data in batch_data:
            dic = {}
            elmo_text.append([char for char in data['text']])
            text = [char2id.get(_char,0) for _char in data['text']] #1,UNK,0 pad
            bichar = [bichar2id.get(_bichar,0) for _bichar in data['bichar']]

            bio = [BIO2id.get(_bio) for _bio in data['bio']]
            # bio = np.expand_dims(bio,axis=-1)
            dic['text'] = text
            dic['bichar'] = bichar
            dic['bio'] = bio

            new_batch_data.append(dic)

        elmo_input = batcher.batch_sentences(elmo_text)

        return new_batch_data,elmo_input
    if mode == 'test':
        for data in batch_data:
            dic = {}
            elmo_text.append([char for char in data['text']])
            text = [char2id.get(_char, 0) for _char in data['text']]  # 1,UNK,0 pad
            bichar=  [bichar2id.get(_bichar,0) for _bichar in data['bichar']]

            dic['text'] = text
            dic['bichar'] = bichar

            new_batch_data.append(dic)

        return new_batch_data,elmo_text

def getSentenceLengths(sentences):
    #返回字典 [len(sentence),idx]
    sentenceLengths = {}
    for idx in range(len(sentences)):
        sentence = sentences[idx]
        if len(sentence) not in sentenceLengths:
            sentenceLengths[len(sentence)] = []
        sentenceLengths[len(sentence)].append(idx)

    return sentenceLengths

def save_result(test_data,test_pred,path):


    for idx in range(len(test_data)):
        test_data[idx]['pred_bio'] = [id2BIO.get(str(bio)) for bio in test_pred[idx]]

    with codecs.open(path, 'w', encoding='utf-8') as fr:
        json.dump(test_data, fr, indent=4, ensure_ascii=False)


def predictLabels( model,elmo_text, data):

    #char-bilstm 只输入char
    sentences = [_data['text'] for _data in data]
    bichars = [_data['bichar'] for _data in data]
    predLabels = [None] * len(sentences)
    sentenceLengths = getSentenceLengths(sentences)

    for indices in tqdm(sentenceLengths.values()):
        nnInput = []

        #输入数据
        inputText = np.array([sentences[idx] for idx in indices])
        inputBichar = np.array([bichars[idx] for idx in indices])
        inputElmo_text = [elmo_text[idx] for idx in indices]
        input_Elmo = batcher.batch_sentences(inputElmo_text)

        nnInput.append(inputText)
        nnInput.append(inputBichar)
        nnInput.append(input_Elmo)

        predictions = model.predict(nnInput, verbose=False)

        predictions = predictions.argmax(axis=-1)  # Predict classes

        predIdx = 0
        for idx in indices:
            predLabels[idx] = predictions[predIdx]
            predIdx += 1

    return predLabels

def pred(model,model_path,result_path,test_data):

    #重载模型
    model.load_weights(model_path)

    _test_data, elmo_text = process_batch_data(test_data, char2id, BIO2id, 'test')
    test_pred = predictLabels(model,elmo_text, _test_data)
    save_result(test_data,test_pred,result_path)


if __name__ == '__main__':

    models=  ['bilstm_cnn','bilstm_selfatt','local_att_gru']
    dims = [200,250,300,'150_300','200_250']
    for model in models:
        for dim in dims:

            if dim == 200:

                elmo_dim = 200
                model_save = '200'
                _params = {'char_embedding_size': 200, 'bichar_embedding_size': 200}
                params.update(_params)
                char_word2vec, char_glove, char_fasttext, bichar_word2vec, bichar_glove, \
                bichar_fasttext = load_embedding(char2id, bichar2id, 200)

            elif dim == 250:

                elmo_dim = 250
                model_save = '250'
                _params = {'char_embedding_size': 250, 'bichar_embedding_size': 250}
                params.update(_params)
                char_word2vec, char_glove, char_fasttext, bichar_word2vec, bichar_glove, \
                bichar_fasttext = load_embedding(char2id, bichar2id, 250)

            elif dim == 300:

                elmo_dim = 300
                model_save = '300'
                _params = {'char_embedding_size': 300, 'bichar_embedding_size': 300}
                params.update(_params)
                char_word2vec, char_glove, char_fasttext, bichar_word2vec, bichar_glove, \
                bichar_fasttext = load_embedding(char2id, bichar2id, 300)

            elif dim == '150_300':

                elmo_dim = 150
                dim = 150
                model_save = '150_300'
                _params = {'char_embedding_size': 150, 'bichar_embedding_size': 300}
                params.update(_params)
                char_word2vec, char_glove, char_fasttext, bichar_word2vec, bichar_glove, \
                bichar_fasttext = load_embedding(char2id, bichar2id, dim=150, dim2=300)

            elif dim == '200_250':

                elmo_dim = 200
                dim = 200
                model_save = '200_250'
                _params = {'char_embedding_size': 200, 'bichar_embedding_size': 250}
                params.update(_params)
                char_word2vec, char_glove, char_fasttext, bichar_word2vec, bichar_glove, \
                bichar_fasttext = load_embedding(char2id, bichar2id, dim=200, dim2=250)

            for cv in range(0,5):
                #模型地址，保存结果地址
                model_path = './models/{}dim/'.format(dim) + model+'{}'.format(cv)+'.weights'
                result_path = './outputs/{}dim/'.format(dim) + model + '{}'.format(cv) +'.json'

                if model == 'bilstm_cnn':
                    pred_model = bilstm_cnn(char_word2vec, char_glove, char_fasttext, bichar_word2vec, bichar_glove, bichar_fasttext, elmo_dim, params)
                    _model = pred_model.build_model()

                elif model == 'bilstm_selfatt':
                    pred_model = bilstm_self_att(char_word2vec, char_glove, char_fasttext, bichar_word2vec, bichar_glove, bichar_fasttext, elmo_dim, params)
                    _model = pred_model.build_model()

                elif model == 'local_att_gru':
                    pred_model = local_att_gru(char_word2vec, char_glove, char_fasttext, bichar_word2vec, bichar_glove, bichar_fasttext, elmo_dim, params)
                    _model = pred_model.build_model()


                pred(_model,model_path,result_path,test_data)
                K.clear_session()