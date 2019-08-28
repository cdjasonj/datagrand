# -*- coding: utf-8 -*-
# 构建供Word2Vec+GLoVe与ELMo使用的预训练语料

import json
import codecs
import gensim
from glove import Glove, Corpus
from gensim.models import word2vec, FastText
from Elmo import train_elmo
from Elmo.training import dump_weights as dw
import h5py

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def ConstructCorpusForWord2Vec():
    originalTrainData = json.load(open('Dataset/json/original_train_data_me.json', encoding='utf-8'))
    testData = json.load(open('Dataset/json/test_data_me.json', encoding='utf-8'))

    textList = []
    for data in originalTrainData:
        text = data['text'].replace(" ", "")
        newText = ""
        for ch in text:
            ch = ch + ' '
            newText += ch
        textList.append(newText)

    for data in testData:
        text = data['text'].replace(" ", "")
        newText = ""
        for ch in text:
            ch = ch + ' '
            newText += ch
        textList.append(newText)

    fr = open('Corpus/CorpusForWord2Vec.txt', 'w', encoding='utf-8')
    for text in textList:
        fr.writelines(text.strip() + '\n')


def preTrainWord2Vec(dimNum, saveFile):
    fr = open('Corpus/processed_corpus.txt', 'r', encoding='utf-8')
    sentenceList = []
    for line in fr.readlines():
        sentenceList.append(line.strip().split(' '))
    word2VecModel = gensim.models.Word2Vec(sentenceList, size=dimNum, sg=1, iter=10, window=10, workers=20)
    word2VecModel.wv.save_word2vec_format(saveFile, binary=False)

    fr.close()


# ———————————————————————————————————GloVe———————————————————————————————————————————
def ConstructCorpusForGloVe():
    fr = open('Corpus/processed_corpus.txt', 'r', encoding='utf-8')
    sentenceList = []
    for line in fr.readlines():
        sentence = line.strip().split(' ')
        sentenceList.append(sentence)
    fr.close()

    corpus_model = Corpus()
    corpus_model.fit(sentenceList, window=10, ignore_missing=False)
    # corpus_model.save('corpus.model')
    print('Dict size: %s' % len(corpus_model.dictionary))
    print('Collocations: %s' % corpus_model.matrix.nnz)

    return corpus_model


def gloveTrans(model):
    modelList = []
    for key in model.dictionary.keys():
        temp = []
        temp.append(key)
        temp += model.word_vectors[model.dictionary[key]].tolist()
        modelList.append(temp)

    return modelList


def preTrainGloVe(dimNum, saveFile):
    corpus_model = ConstructCorpusForGloVe()
    # corpus_model.save('Corpus/CorpusForDaGuanGlove_200dim.model')
    glove = Glove(no_components=dimNum, learning_rate=0.05)
    glove.fit(corpus_model.matrix, epochs=10, no_threads=20, verbose=True)
    glove.add_dictionary(corpus_model.dictionary)
    gloveModel = gloveTrans(glove)
    with open(saveFile, 'w', encoding='utf-8') as fr1:
        for temp in gloveModel:
            s = ''
            for ch in temp:
                s = s + str(ch) + ' '
            fr1.writelines(s.strip() + '\n')
    fr1.close()


# ———————————————————————————————————ELMo———————————————————————————————————————————
def ConstructCorpusForELMo():
    '''
    originalTrainData=json.load(open('Dataset/json/original_train_data_me.json',encoding='utf-8'))
    testData=json.load(open('Dataset/json/test_data_me.json',encoding='utf-8'))
    corpusData=originalTrainData+testData

    #按词频构建词典
    charsVocab={}
    for data in corpusData:
        for ch in data['text']:
            charsVocab[ch]=charsVocab.get(ch,0)+1
    charsVocab=sorted(charsVocab.items(),key = lambda x:x[1],reverse = True)

    charsList=['<S>','</S>','<UNK>']
    for temp in charsVocab:
        charsList.append(temp[0])

    fr=open('Corpus/VocabForElmo.txt','w',encoding='utf-8')
    for char in charsList:
        fr.writelines(char+'\n')
    '''

    # 按词频构建词典
    fr = open('Corpus/DaGuan/processed_corpus.txt', 'r', encoding='utf-8')
    charsVocab = {}
    for line in fr.readlines():
        temp = line.strip().split(' ')
        for ch in temp:
            charsVocab[ch] = charsVocab.get(ch, 0) + 1
    charsVocab = sorted(charsVocab.items(), key=lambda x: x[1], reverse=True)  # 返回的是一个list，list里面每个元素是一个tuple，tuple中第一个元素是char，第二个元素是词频

    charsList = ['<S>', '</S>', '<UNK>']
    for temp in charsVocab:
        charsList.append(temp[0])

    if '' in charsList:  # 原先词典中的5622是个''
        print(charsList.index(''))
        charsList.remove('')

    fr = open('Corpus/DaGuan/DaGuanVocabForElmo.txt', 'w', encoding='utf-8')
    for char in charsList:
        fr.writelines(char + '\n')


def ConstructBicharCorpusForElmo():
    fr1 = open('Corpus/DaGuan/processed_corpus.txt', 'r', encoding='utf-8')
    sentenceList = []
    for line in fr1.readlines():
        new_text = []
        temp = line.strip().split(' ')
        for index, char in enumerate(temp):
            if index != len(temp) - 1:  # 没有达到最后一个
                new_text.append(char + '_' + temp[index + 1])
            else:
                new_text.append(char + '_$')
        sentenceList.append(new_text)

    fr2 = open('Corpus/DaGuan/processed_corpus_bichar.txt', 'w', encoding='utf-8')
    fr3 = open('Corpus/DaGuan/DaGuanBicharVocabForElmo_mincount_5.txt', 'w', encoding='utf-8')
    bicharVocab = {}
    for sentence in sentenceList:
        fr2.write(' '.join(sentence) + '\n')
        for ch in sentence:
            bicharVocab[ch] = bicharVocab.get(ch, 0) + 1
    bicharVocab = sorted(bicharVocab.items(), key=lambda x: x[1], reverse=True)
    bicharList = ['<S>', '</S>', '<UNK>']
    for temp in bicharVocab:
        if temp[1] >= 5:
            bicharList.append(temp[0])
    if '' in bicharList:
        bicharList.remove('')
    print('the length of bicharVocab is {}'.format(len(bicharList)))  # no word count:1504660,min_count_5:714297(714294+3)
    for bichar in bicharList:
        fr3.writelines(bichar + '\n')


def preTrainELMo(mode):
    if mode == 'char':
        vocabPath = 'Corpus/DaGuanVocabForElmo.txt'
        savePath = 'CharEmbedding/checkpointForElmo_200dim'
        trainPrefix = 'Corpus/processed_corpus.txt'
        train_elmo.preTrain(vocabPath, savePath, trainPrefix, trainCharsNum=93245582)
    elif mode == 'bichar':
        vocabPath = 'Corpus/DaGuan/DaGuanBicharVocabForElmo_mincount_5.txt'
        savePath = 'CharEmbedding/DaGuan/checkpointForBicharElmo'
        trainPrefix = 'Corpus/DaGuan/processed_corpus_bichar.txt'
        train_elmo.preTrain(vocabPath, savePath, trainPrefix, trainCharsNum=93245582)


# ——————————————————————————————————————FastText———————————————————————————————————————
def fastTextTrans(model):
    modelList = []
    for key in model.wv.vocab.keys():
        temp = []
        temp.append(key)
        temp += model[key].tolist()
        modelList.append(temp)

    return modelList


def preTrainFastText(dimNum, saveFile):
    fr = open('Corpus/processed_corpus.txt', 'r', encoding='utf-8')
    sentenceList = []
    for line in fr.readlines():
        sentenceList.append(line.strip().split(' '))

    model = FastText(sentenceList, size=dimNum, window=10, workers=20, sg=1, iter=10, min_n=2, max_n=8)
    fastTextModel = fastTextTrans(model)
    with open(saveFile, 'w', encoding='utf-8') as fr1:
        for temp in fastTextModel:
            s = ''
            for ch in temp:
                s = s + str(ch) + ' '
            fr1.write(s.strip() + '\n')
    fr.close()
    fr1.close()


# ——————————————————————————————————————DaGuan语料构建———————————————————————————————————————
def constructCorpusForDaGuan():
    corpus = []
    with open('Corpus/DaGuan/corpus.txt', encoding='utf-8') as fr:
        for line in fr:
            corpus.append(line.strip().split('_'))

    fr.close()
    with open('Corpus/DaGuan/processed_corpus.txt', 'w', encoding='utf-8') as fr:
        for c in corpus:
            fr.write(' '.join(c) + '\n')


# ——————————————————————————————————————DaGuan杯预训练———————————————————————————————————————
def getPreTrain(sentenceList, mode, dimNum, minCount, saveFileWord2Vec, saveFileFastText, saveFileGlove=''):  # 从语料中构建bichar形式的list，并预训练bichar的word2vec，glove，FastText
    '''
    gloveCorpusModel=Corpus()
    gloveCorpusModel.fit(sentenceList, window=10,ignore_missing=False)
    #corpus_model.save('corpus.model')
    print('Dict size: %s' % len(gloveCorpusModel.dictionary))
    print('Collocations: %s' % gloveCorpusModel.matrix.nnz)
    GloveModel = Glove(no_components=300, learning_rate=0.05)
    GloveModel.fit(gloveCorpusModel.matrix, epochs=25,no_threads=20, verbose=True)
    GloveModel.add_dictionary(gloveCorpusModel.dictionary)
    glove=gloveTrans(GloveModel)
    if mode=='bichar':
        with open('CharEmbedding/DaGuan/DaGuanBicharGlove2.0.txt','w',encoding='utf-8') as fr1:
            for temp in glove:
                s=''
                for ch in temp:
                    s=s+str(ch)+' '
                fr1.writelines(s.strip()+'\n')
        fr1.close()

    elif mode=='trichar':
        with open('CharEmbedding/DaGuan/DaGuanTricharGlove1.0.txt','w',encoding='utf-8') as fr1:
            for temp in glove:
                s=''
                for ch in temp:
                    s=s+str(ch)+' '
                fr1.writelines(s.strip()+'\n')
        fr1.close()
    '''

    Word2VecModel = gensim.models.Word2Vec(sentenceList, size=dimNum, sg=1, iter=15, window=10, workers=20, min_count=minCount)  # 上下文窗口默认为5
    if mode == 'bichar':
        Word2VecModel.wv.save_word2vec_format(saveFileWord2Vec, binary=False)
    elif mode == 'trichar':
        Word2VecModel.wv.save_word2vec_format(saveFileWord2Vec, binary=False)

    FastTextModel = FastText(sentenceList, size=dimNum, window=10, workers=20, sg=1, iter=15, min_n=2, max_n=8, min_count=minCount)
    fastText = fastTextTrans(FastTextModel)
    if mode == 'bichar':
        with open(saveFileFastText, 'w', encoding='utf-8') as fr2:
            for temp in fastText:
                s = ''
                for ch in temp:
                    s = s + str(ch) + ' '
                fr2.write(s.strip() + '\n')
        fr2.close()
    elif mode == 'trichar':
        with open(saveFileFastText, 'w', encoding='utf-8') as fr2:
            for temp in fastText:
                s = ''
                for ch in temp:
                    s = s + str(ch) + ' '
                fr2.write(s.strip() + '\n')
        fr2.close()


def createSentenceList(mode):
    fr1 = open('Corpus/processed_corpus.txt', 'r', encoding='utf-8')
    # fr2=open('Corpus/DaGuan/trainCorpus.txt','r',encoding='utf-8')
    # fr3=open('Corpus/DaGuan/testCorpus.txt','r',encoding='utf-8')

    if mode == 'char':
        sentenceList = []
        for line in fr1.readlines():
            sentenceList.append(line.strip().split(' '))
        '''
        for line in fr2.readlines():
            sentenceList.append(line.strip().split(' '))
        for line in fr3.readlines():
            sentenceList.append(line.strip().split(' '))
        '''
        fr1.close()
        return sentenceList

    elif mode == 'bichar':
        sentenceList = []
        for line in fr1.readlines():
            new_text = []
            temp = line.strip().split(' ')
            for index, char in enumerate(temp):
                if index != len(temp) - 1:  # 没有达到最后一个
                    new_text.append(char + '_' + temp[index + 1])
                else:
                    new_text.append(char + '_$')
            sentenceList.append(new_text)
        '''    
        for line in fr2.readlines():
            new_text=[]
            temp=line.strip().split(' ')
            for index,char in enumerate(temp):
                if index != len(temp) - 1:  # 没有达到最后一个
                    new_text.append(char + '_'+temp[index + 1])
                else:
                    new_text.append(char + '_$')
            sentenceList.append(new_text)

        for line in fr3.readlines():
            new_text=[]
            temp=line.strip().split(' ')
            for index,char in enumerate(temp):
                if index != len(temp) - 1:  # 没有达到最后一个
                    new_text.append(char + '_'+temp[index + 1])
                else:
                    new_text.append(char + '_$')
            sentenceList.append(new_text)
        '''
        fr1.close()
        return sentenceList

    elif mode == 'trichar':
        sentenceList = []
        for line in fr1.readlines():
            new_text = []
            temp = line.strip().split(' ')
            for index, char in enumerate(temp):
                if index < len(temp) - 2:
                    new_text.append(char + '_' + temp[index + 1] + '_' + temp[index + 2])
                elif index == len(temp) - 2:
                    new_text.append(char + '_' + temp[index + 1] + '$')
                elif index == len(temp) - 1:
                    new_text.append(char + '_$_$')
            sentenceList.append(new_text)
        '''
        for line in fr2.readlines():
            new_text=[]
            temp=line.strip().split(' ')
            for index,char in enumerate(temp):
                if index <len(temp)-2:
                    new_text.append(char+'_'+temp[index+1]+'_'+temp[index+2])
                elif index==len(temp)-2:
                    new_text.append(char+'_'+temp[index+1]+'$')
                elif index==len(temp)-1:
                    new_text.append(char+'_$_$')
            sentenceList.append(new_text)

        for line in fr3.readlines():
            new_text=[]
            temp=line.strip().split(' ')
            for index,char in enumerate(temp):
                if index <len(temp)-2:
                    new_text.append(char+'_'+temp[index+1]+'_'+temp[index+2])
                elif index==len(temp)-2:
                    new_text.append(char+'_'+temp[index+1]+'$')
                elif index==len(temp)-1:
                    new_text.append(char+'_$_$')
            sentenceList.append(new_text)
        '''
        fr1.close()
        return sentenceList


def trainAndTestProcess():
    fr1 = open('Dataset/DaGuan/train.txt', 'r', encoding='utf-8')
    fr2 = open('Corpus/DaGuan/trainCorpus.txt', 'w', encoding='utf-8')
    for line in fr1.readlines():
        line = line.strip()
        line = line.replace('/a  ', '_')
        line = line.replace('/b  ', '_')
        line = line.replace('/c  ', '_')
        line = line.replace('/o  ', '_')
        line = line.replace('/a', '')
        line = line.replace('/b', '')
        line = line.replace('/c', '')
        line = line.replace('/o', '')
        temp = line.split('_')
        s = ''
        for ch in temp:
            s = s + ch + ' '
        fr2.write(s.strip() + '\n')

    fr3 = open('Dataset/DaGuan/test.txt', 'r', encoding='utf-8')
    fr4 = open('Corpus/DaGuan/testCorpus.txt', 'w', encoding='utf-8')
    for line in fr3.readlines():
        temp = line.strip().split('_')
        s = ''
        for ch in temp:
            s = s + ch + ' '
        fr4.write(s.strip() + '\n')

    fr1.close()
    fr2.close()
    fr3.close()
    fr4.close()


# ——————————————————————————————————————查看一些信息———————————————————————————————————————
def infoForDaGuan():  # 查看语料的总字数(去重复)，以及预训练的模型的字典在训练集中的覆盖率
    fr = open('Corpus/DaGuan/processed_corpus.txt', 'r', encoding='utf-8')
    charsCorpus = []
    for line in fr.readlines():
        charsCorpus += line.strip().split(' ')
    print('预训练语料总字数为:{}'.format(len(charsCorpus)))  # 93245582
    charsCorpus = list(set(charsCorpus))
    print('预训练语料字数(去重复)为:{}'.format(len(charsCorpus)))  # 21148 有''??? 实际数量应该是21147

    fr1 = open('Corpus/DaGuan/processed_corpus_bichar.txt', 'r', encoding='utf-8')
    bicharsCorpus = []
    for line in fr1.readlines():
        bicharsCorpus += line.strip().split(' ')
    print('bichar预训练语料总字数为:{}'.format(len(bicharsCorpus)))  # 93245582
    bicharsCorpus = list(set(bicharsCorpus))
    print('预训练语料字数(去重复)为:{}'.format(len(bicharsCorpus)))  # 1504657

    fr2 = open('Corpus/DaGuan/processed_corpus_bichar_debug.txt', 'r', encoding='utf-8')
    bicharsCorpusDebug = []
    for line in fr2.readlines():
        bicharsCorpusDebug += line.strip().split(' ')
    print('bichar预训练debug语料总字数为:{}'.format(len(bicharsCorpusDebug)))  # 53897
    bicharsCorpusDebug = list(set(bicharsCorpusDebug))
    print('预训练debug语料字数(去重复)为:{}'.format(len(bicharsCorpusDebug)))  # 46170

    fr.close()
    fr1.close()

    '''
    fr1=open('Dataset/DaGuan/train.txt','r',encoding='utf-8')    
    charsTrain=[]
    for line in fr1.readlines():
        line=line.replace('/a  ','_')
        line=line.replace('/b  ','_')
        line=line.replace('/c  ','_')
        line=line.replace('/o  ','_')
        line=line.replace('/a','')
        line=line.replace('/b','')
        line=line.replace('/c','')
        line=line.replace('/o','')
        charsTrain+=line.split('_')
    charsTrain=list(set(charsTrain))

    fr2=open('CharEmbedding/DaGuan/DaGuanWord2Vec.txt','r',encoding='utf-8')
    charsWord2Vec=[]
    temp=fr2.readlines()
    for i in range(1,len(temp)):
        charsWord2Vec.append(temp[i].split(' ')[0])    
    result2=list(set(charsWord2Vec).intersection(set(charsTrain)))
    print('word2vec字典的字数为:{}'.format(len(charsWord2Vec)))
    print('训练集字数为:{}'.format(len(charsTrain)))
    print('word2vec覆盖训练集的比例为:{}'.format(len(result2) / len(charsTrain)))

    DaGuanGlove=Glove.load('CharEmbedding/DaGuan/DaGuanGlove.model')
    charsGlove=DaGuanGlove.dictionary.keys()
    print('Glove字典的字数为:{}'.format(len(charsGlove)))
    result3=list(set(charsGlove).intersection(set(charsTrain)))
    print('Glove覆盖训练集的比例为:{}'.format(len(result3) / len(charsTrain)))
    #结果：预训练语料有21278个字，训练集有4653个字，word2vec进行了词频小于5的过滤，有9996个字，覆盖了训练集的95.3%，glove没有根据词频进行过滤，有21278个字，覆盖了训练集的97.3%
    '''


'''
sentenceListBichar=createSentenceList('bichar')
getPreTrain(sentenceListBichar,'bichar',5)
print('bichar已train好')
sentenceListTrichar=createSentenceList('trichar')
getPreTrain(sentenceListTrichar,'trichar',7)
'''
# dw('CharEmbedding/checkpointForElmo_200dim','CharEmbedding/checkpointForElmo_200dim/weights.hdf5')
# constructCorpusForDaGuan()

'''            
preTrainGloVe(dimNum=50,saveFile='CharEmbedding/DaGuanGlove_50dim.txt')  
print('50维的Glove已pretrain好')
preTrainGloVe(dimNum=150,saveFile='CharEmbedding/DaGuanGlove_150dim.txt')  
print('150维的Glove已pretrain好')
preTrainGloVe(dimNum=200,saveFile='CharEmbedding/DaGuanGlove_200dim.txt')  
print('200维的Glove已pretrain好')


preTrainWord2Vec(dimNum=50,saveFile='CharEmbedding/DaGuanWord2Vec_50dim.txt')
print('50维的word2vec已pretrain好')
preTrainWord2Vec(dimNum=150,saveFile='CharEmbedding/DaGuanWord2Vec_150dim.txt')
print('150维的word2vec已pretrain好')
preTrainWord2Vec(dimNum=200,saveFile='CharEmbedding/DaGuanWord2Vec_200dim.txt')
print('200维的word2vec已pretrain好')

preTrainFastText(dimNum=50,saveFile='CharEmbedding/DaGuanFastText_50dim.txt')
print('50维的fastText已pretrain好')
preTrainFastText(dimNum=150,saveFile='CharEmbedding/DaGuanFastText_150dim.txt')
print('150维的fastText已pretrain好')
preTrainFastText(dimNum=200,saveFile='CharEmbedding/DaGuanFastText_200dim.txt')
print('200维的fastText已pretrain好')

sentenceList=createSentenceList(mode='bichar')
getPreTrain(sentenceList,mode='bichar',dimNum=50,minCount=5,saveFileWord2Vec='CharEmbedding/DaGuanBicharWord2Vec_50dim.txt',saveFileFastText='CharEmbedding/DaGuanBicharFastText_50dim.txt')
print('50维的bichar已train好')
getPreTrain(sentenceList,mode='bichar',dimNum=150,minCount=5,saveFileWord2Vec='CharEmbedding/DaGuanBicharWord2Vec_150dim.txt',saveFileFastText='CharEmbedding/DaGuanBicharFastText_150dim.txt')
print('150维的bichar已train好')
getPreTrain(sentenceList,mode='bichar',dimNum=200,minCount=5,saveFileWord2Vec='CharEmbedding/DaGuanBicharWord2Vec_200dim.txt',saveFileFastText='CharEmbedding/DaGuanBicharFastText_200dim.txt')
print('200维的bichar已train好')




'''

getPreTrain(sentenceList, mode='bichar', dimNum=250, minCount=5, saveFileWord2Vec='CharEmbedding/DaGuanBicharWord2Vec_250dim.txt',
            saveFileFastText='CharEmbedding/DaGuanBicharFastText_250dim.txt')
print('200维的bichar已train好')

# preTrainELMo('char')

# ConstructBicharCorpusForElmo()
# infoForDaGuan()

# trainAndTestProcess()
'''
fr=open('Corpus/DaGuan/processed_corpus_bichar.txt','r',encoding='utf-8')
temp=fr.readlines()
debug_temp=temp[:1000]
fr1=open('Corpus/DaGuan/processed_corpus_bichar_debug.txt','w',encoding='utf-8')
for line in debug_temp:
    fr1.write(line.strip()+'\n')
fr.close()
fr1.close() 
'''

'''
model = FastText.load('CharEmbedding/DaGuan/OriginalDaGuanFastText.txt')
print(type(model))
print(model.wv.get_vector('15274'))
print(model['15274'])
print(model.wv.get_vector('15274\n'))
print(model['15274\n'])
print(type(model.wv.vocab))
modelList=[]
for key in model.wv.vocab.keys():
    temp=[]
    if '\n' not in key:
        temp.append(key.strip())
        temp+=model[key].tolist()
        modelList.append(temp)
with open('CharEmbedding/DaGuan/DaGuanFastText.txt','w',encoding='utf-8') as fr1:    
    for temp in modelList:
        s=''
        for ch in temp:
            s=s+str(ch)+' '
        fr1.write(s.strip()+'\n')
fr1.close()
'''

'''
gloveModel=Glove.load('CharEmbedding/DaGuan/DaGuanGlove.model')
print(type(gloveModel.word_vectors[gloveModel.dictionary['国']]))
print(gloveModel.word_vectors[gloveModel.dictionary['国']])
'''

'''
fr=open('CharEmbedding/Word2Vec.txt','r',encoding='utf-8')
for line in fr.readlines():
	temp=line.split(' ')
	if temp[0]=='国':
		vector=[]
		for i in range(1,101):
			vector.append(float(temp[i]))
		break
print(vector)
'''
'''
fr1=open('CharEmbedding/DaGuan/DaGuanBicharGloveForElmo.txt','r',encoding='utf-8')
temp=fr1.readlines()
print(len(temp))
'''
#
# fr = h5py.File('CharEmbedding/ELMo/150dim/DaGuanElmo_150dim.hdf5', 'r')
# fr1 = open('CharEmbedding/ELMo/150dim/DaGuanElmoPretrainEmbedding_150dim.txt', 'w', encoding='utf-8')
# fr2 = open('Corpus/DaGuanVocabForElmo.txt', 'r', encoding='utf-8')
# tokenList = fr2.readlines()
# for i in range(len(fr['embedding'])):
#     embedding = fr['embedding'][i].tolist()
#     embedding = list(map(str, embedding))
#     token = tokenList[i].strip()
#     fr1.write(token + ' ')
#     fr1.write(' '.join(embedding) + '\n')



