import json
import math
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import KFold
import codecs
from tools.conlleval import  evaluate_conll_file
from keras import backend as K
from tools.load_embedding_matrix import load_embedding
from bilm import TokenBatcher
from inputs.data_process import collect_entities,load_AG_data
import random
from neuralnets.ELMo.elmo_bilstm_selatt_crf import Bilstm_selfatt_crf

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

train_data = json.load(open('./inputs/train_data.json',encoding='utf-8'))
test_data = json.load(open('./inputs/test_data.json',encoding='utf-8'))
id2char, char2id = json.load(open('./inputs/char2id.json', encoding='utf-8'))
id2bichar, bichar2id = json.load(open('./inputs/bichar2id.json', encoding='utf-8'))
id2BIO, BIO2id = json.load(open('./inputs/bio2id.json', encoding='utf-8'))

params = {'char2id_size':len(char2id),'char_embedding_size':150,'epochs':100,'bichar_embedding_size':150,
          'early_stopping':8,'bichar2id_size':len(bichar2id)
          ,'n_class_labels':len(BIO2id),'model_save_path': './models/bilstm_selfatt{}.weights'
          ,}

is_use_self_training = False
debug = False
if debug:
    train_data = train_data[:200]

vocab_file = './ELMo/DaGuanVocabForElmo.txt'
batcher = TokenBatcher(vocab_file)

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


def minibatch_iterate_dataset(trainData,miniBatchSize=128):

    trainData.sort(key=lambda x: len(x['text']))  # Sort train matrix by sentence length
    trainRanges = []
    oldSentLength = len(trainData[0]['text'])
    idxStart = 0
    #Shuffle TrainData
    # Find start and end of ranges with sentences with same length
    for idx in range(len(trainData)):
        sentLength = len(trainData[idx]['text'])

        if sentLength != oldSentLength:
            trainRanges.append((idxStart, idx))
            idxStart = idx

        oldSentLength = sentLength

    # Add last sentence
    trainRanges.append((idxStart, len(trainData)))

    # Break up ranges into smaller mini batch sizes
    miniBatchRanges = []
    for batchRange in trainRanges:
        rangeLen = batchRange[1] - batchRange[0]
        bins = int(math.ceil(rangeLen / float(miniBatchSize)))
        binSize = int(math.ceil(rangeLen / float(bins)))

        for binNr in range(bins):
            startIdx = binNr * binSize + batchRange[0]
            endIdx = min(batchRange[1], (binNr + 1) * binSize + batchRange[0])
            miniBatchRanges.append((startIdx, endIdx))

    #shuffle minBatchRanges
    np.random.shuffle(miniBatchRanges)

    for miniRange in tqdm(miniBatchRanges):
        # print(miniRange)
        batch_data = []
        for i in range(miniRange[0],miniRange[1]):
            batch_data.append(trainData[i])
            #序列化text,label
        batch_data,elmo_input = process_batch_data(batch_data,char2id,BIO2id,'dev')

        yield batch_data,elmo_input


def trainModel(model,train_data):
    for batch,elmo_input in minibatch_iterate_dataset(train_data):
        Inputs = []
        Labels = np.array([data['bio'] for data in batch])
        Inputs_text = np.array([data['text'] for data in batch])
        Inputs_bichar = np.array([data['bichar'] for data in batch])

        Inputs.append(Inputs_text)
        Inputs.append(Inputs_bichar)
        Inputs.append(elmo_input)

        Labels = np.expand_dims(Labels,axis=-1)
        model.train_on_batch(Inputs, Labels)


def getSentenceLengths(sentences):
    #返回字典 [len(sentence),idx]
    sentenceLengths = {}
    for idx in range(len(sentences)):
        sentence = sentences[idx]
        if len(sentence) not in sentenceLengths:
            sentenceLengths[len(sentence)] = []
        sentenceLengths[len(sentence)].append(idx)

    return sentenceLengths

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


#ToDO
def compute_f1(dev_data,dev_pred):

    def save_result(ner_pred, ner_true, save_file):
        """
        保存 text ture pred
        ner_pred: [[00021000..],...[]]
        ner_true: json [text:; bio:]
        :return:
        """

        pred_bio = []
        for _pred_bio in ner_pred:
            temp = [id2BIO.get(str(bio)) for bio in _pred_bio]
            pred_bio.append(temp)

        with open(save_file, 'w', encoding='utf-8') as fr:
            for idx in range(len(pred_bio)):
                for i in range(len(ner_true[idx]['text'])):
                    fr.write(str(ner_true[idx]['text'][i]) + ' ' + str(ner_true[idx]['bio'][i]) + ' ' + str(pred_bio[idx][i]) + '\n')

    save_result(dev_pred,dev_data,'result_for_f1_elmo_selfatt')
    p,r,f = evaluate_conll_file('result_for_f1_elmo_selfatt')

    return p,r,f

#ToDo 按照json保存， 还是在实体级别上进行投票， 然后在后处理对bio按照实体进行转换。
def save_result(test_data,test_pred,path):


    for idx in range(len(test_data)):
        test_data[idx]['pred_bio'] = [id2BIO.get(str(bio)) for bio in test_pred[idx]]

    with codecs.open(path, 'w', encoding='utf-8') as fr:
        json.dump(test_data, fr, indent=4, ensure_ascii=False)

def fit(params,elmo_dim,train_data,dev_data,test_data,cv_num,char_word2vec,char_glove,char_fasttext,bichar_word2vec,bichar_glove,bichar_fastext):
    epochs = params['epochs']
    bilstm = Bilstm_selfatt_crf(char_word2vec,char_glove,char_fasttext,bichar_word2vec,bichar_glove,bichar_fastext,elmo_dim,params)
    model = bilstm.build_model()
    best_dev_p, best_dev_r, best_dev_f = 0,0,0

    patient = 0
    lr_patient = 0
    for epoch in range(epochs):
        trainModel(model,train_data)
        _dev_data, elmo_text = process_batch_data(dev_data, char2id, BIO2id, 'test')
        dev_pred = predictLabels(model,elmo_text,_dev_data)

        p,r,f = compute_f1(dev_data,dev_pred)
        # save_result(dev_pred,dev_data,'./outputs/dev')
        # dev_p, dev_r, dev_f = evaluate_conll_file('./outputs/dev')


        print('NER,当前第{}个epoch，测试集,准确度为{},召回为{},f1为：{}'.format(epoch,  p, r, f))
        print('-' * 20)

        if best_dev_f <  f:
            best_dev_p,best_dev_r,best_dev_f = p,r,f
            model.save_weights(params['model_save_path'].format(cv_num))
            patient = 0
            lr_patient = 0
        else:
            patient += 1
            lr_patient +=1
            if patient == params['early_stopping'] and epoch>= 25 :
                print('当前为{}epoch,触发早停'.format(epoch))
                break

            if lr_patient >= 2:
                #当f1在两个epoch没有提升，那么降低学习率为原来的80%
                current_lr = K.get_value(model.optimizer.lr)
                K.set_value(model.optimizer.lr,0.9 * current_lr)
                lr_patient = 0
                print('当前lr为{} '.format(K.get_value(model.optimizer.lr)))

    print('训练结束')
    print('当前最好的 测试集,准确度为{},召回为{},f1为：{}'.format(best_dev_p,best_dev_r,best_dev_f ))
    _test_data,elmo_text = process_batch_data(test_data, char2id, BIO2id, 'test')
    test_pred = predictLabels(model,elmo_text, _test_data)

    save_result(test_data,test_pred,params['result_save_path'].format(cv_num))

    return test_pred #返回概率

dims = [200,250,300,'200_250','150_300']

for dim in dims:
    char_word2vec,char_glove,char_fasttext,bichar_word2vec,bichar_glove,bichar_fasttext=0,0,0,0,0,0


    if dim == 200:
        _params = {'char_embedding_size':200,'bichar_embedding_size':200,'result_save_path':'./outputs/ELMo/200dim/bilstm_selfatt{}.json'}
        params.update(_params)
        char_word2vec, char_glove, char_fasttext, bichar_word2vec, bichar_glove,\
        bichar_fasttext = load_embedding(char2id,bichar2id,200)

    elif dim == 250:
        _params = {'char_embedding_size':250,'bichar_embedding_size':250,'result_save_path':'./outputs/ELMo/250dim/bilstm_selfatt{}.json'}
        params.update(_params)
        char_word2vec, char_glove, char_fasttext, bichar_word2vec, bichar_glove,\
        bichar_fasttext = load_embedding(char2id,bichar2id,250)

    elif dim == 300:
        _params = {'char_embedding_size':300,'bichar_embedding_size':300,'result_save_path':'./outputs/ELMo/300dim/bilstm_selfatt{}.json'}
        params.update(_params)
        char_word2vec, char_glove, char_fasttext, bichar_word2vec, bichar_glove,\
        bichar_fasttext = load_embedding(char2id,bichar2id,300)

    elif dim == '150_300':

        dim = 150
        _params = {'char_embedding_size':150,'bichar_embedding_size':300,'result_save_path': './outputs/ELMo/150_300dim/bilstm_selfatt{}.json'}
        params.update(_params)
        char_word2vec, char_glove, char_fasttext, bichar_word2vec, bichar_glove, \
        bichar_fasttext = load_embedding(char2id, bichar2id, dim = 150,dim2 = 300)

    elif dim == '200_250':

        dim = 200
        _params = {'char_embedding_size':200,'bichar_embedding_size':250,'result_save_path': './outputs/ELMo/200_250dim/bilstm_selfatt{}.json'}
        params.update(_params)
        char_word2vec, char_glove, char_fasttext, bichar_word2vec, bichar_glove, \
        bichar_fasttext = load_embedding(char2id, bichar2id, dim = 200,dim2 = 250)

    splits = list(KFold(n_splits=5,shuffle=True,random_state=2018).split(train_data))
    cv_test_pred=[]
    for idx, (train_index,dev_index) in enumerate(splits):
        if idx !=3 :
            continue
        # config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        # session = tf.Session(config=config)
        # KTF.set_session(session)
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # session = tf.Session(config=config)
        # KTF.set_session(session)
        print('当前是第{}个cv'.format(idx))
        cv_train_data = [train_data[_idx] for _idx in train_index]
        entities = collect_entities(list(train_index))  # 收集数据增强的实体，只在train_index收集，防止数据泄露
        AG_data = load_AG_data(entities)  # 生成数据增强数据
        AG_index = random.sample(list(train_index),8000)
        # 只用train_index生成的数据增强部分
        _AG_data = [AG_data[_idx] for _idx in AG_index]
        cv_train_data += _AG_data

        cv_dev_data = [train_data[_idx] for _idx in dev_index]
        cv_test_pred.append(fit(params,dim,cv_train_data,cv_dev_data,test_data,idx,char_word2vec,char_glove,char_fasttext,
                                bichar_word2vec,bichar_glove,bichar_fasttext))
        K.clear_session()
