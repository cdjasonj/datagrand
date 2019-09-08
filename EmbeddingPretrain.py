# -*- coding: utf-8 -*-
import gensim
from gensim.models import word2vec,FastText
from glove import Glove,Corpus
from ElmoSourceCode import train_elmo
from ElmoSourceCode.training import dump_weights as dw
import h5py
import shutil

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#——————————————————————————————————————构建用于预训练的语料———————————————————————————————————————     
def constructCorpusForDaGuan():  #将主办方提供的语料处理成token之间用空格隔开的形式
    corpus = []
    with open('inputs/corpus/corpus.txt',encoding='utf-8') as fr:
        for line in fr:
            corpus.append(line.strip().split('_'))

    fr.close()
    with open('inputs/corpus/processed_corpus.txt','w',encoding='utf-8') as fr:
        for c in corpus:
            fr.write(' '.join(c) + '\n')

def createSentenceList(mode):  #该函数将语料处理成sentencelist，形式为[[token1,token2,token3...],[token1,token2,token3...]....],可以选择char、bichar或trichar，trichar效果不佳
    fr1=open('inputs/corpus/processed_corpus.txt','r',encoding='utf-8')
    
    if mode=='char':
        sentenceList=[]
        for line in fr1.readlines():
            sentenceList.append(line.strip().split(' '))
        fr1.close()
        return sentenceList
        
    elif mode=='bichar':
        sentenceList=[]
        for line in fr1.readlines():
            new_text=[]
            temp=line.strip().split(' ')
            for index,char in enumerate(temp):
                if index != len(temp) - 1:  # 没有达到最后一个
                    new_text.append(char + '_'+temp[index + 1])
                else:
                    new_text.append(char + '_$')
            sentenceList.append(new_text)
        fr1.close()
        return sentenceList

       
#——————————————————————————————————————DaGuan杯word2vec,glove,fastText预训练———————————————————————————————————————     
def fastTextTrans(model):
    modelList=[]
    for key in model.wv.vocab.keys():
        temp=[]
        temp.append(key)
        temp+=model[key].tolist()
        modelList.append(temp)
    
    return modelList

def gloveTrans(model):
    modelList=[]
    for key in model.dictionary.keys():
        temp=[]
        temp.append(key)
        temp+=model.word_vectors[model.dictionary[key]].tolist()
        modelList.append(temp)
    
    return modelList

def getPreTrain(mode,dimNum,minCount,saveFileWord2Vec,saveFileFastText,saveFileGlove):#从语料中构建bichar形式的list，并预训练bichar的word2vec，glove，FastText
    sentenceList=createSentenceList(mode)
    
    gloveCorpusModel=Corpus()
    gloveCorpusModel.fit(sentenceList, window=10,ignore_missing=False)
    #corpus_model.save('corpus.model')
    print('Dict size: %s' % len(gloveCorpusModel.dictionary))
    print('Collocations: %s' % gloveCorpusModel.matrix.nnz)
    GloveModel = Glove(no_components=300, learning_rate=0.05)
    GloveModel.fit(gloveCorpusModel.matrix, epochs=25,no_threads=20, verbose=True)
    GloveModel.add_dictionary(gloveCorpusModel.dictionary)
    glove=gloveTrans(GloveModel)
    with open(saveFileGlove,'w',encoding='utf-8') as fr1:
        for temp in glove:
            s=''
            for ch in temp:
                s=s+str(ch)+' '            
            fr1.writelines(s.strip()+'\n')
    fr1.close()
    
    
    Word2VecModel=gensim.models.Word2Vec(sentenceList,size=dimNum,sg=1,iter=15,window=10,workers=20,min_count=minCount) #上下文窗口默认为5
    Word2VecModel.wv.save_word2vec_format(saveFileWord2Vec,binary=False)
    
    FastTextModel=FastText(sentenceList,size=dimNum,window=10,workers=20,sg=1,iter=15,min_n=2,max_n=8,min_count=minCount)
    fastText=fastTextTrans(FastTextModel)
    with open(saveFileFastText,'w',encoding='utf-8') as fr2:
        for temp in fastText:
            s=''
            for ch in temp:
                s=s+str(ch)+' '            
            fr2.write(s.strip()+'\n')
    fr2.close()

#———————————————————————————————————DaGuan杯ELMo预训练———————————————————————————————————————————    
def ConstructCorpusForELMo(): 
    #-----------------按词频构建char的词典----------------
    fr=open('inputs/corpus/processed_corpus.txt','r',encoding='utf-8')
    charsVocab={}
    for line in fr.readlines():
        temp=line.strip().split(' ')
        for ch in temp:
            charsVocab[ch]=charsVocab.get(ch,0)+1
    charsVocab=sorted(charsVocab.items(),key = lambda x:x[1],reverse = True) #返回的是一个list，list里面每个元素是一个tuple，tuple中第一个元素是char，第二个元素是词频
    
    charsList=['<S>','</S>','<UNK>']
    for temp in charsVocab:
        charsList.append(temp[0])
    
    if '' in charsList:   #原先词典中的5622是个''
        print(charsList.index(''))
        charsList.remove('')
    
    fr=open('inputs/corpus/DaGuanVocabForElmo.txt','w',encoding='utf-8')
    for char in charsList:
        fr.writelines(char+'\n')
    
    shutil.copy('inputs/corpus/DaGuanVocabForElmo.txt','ELMo/150dim/DaGuanVocabForElmo.txt')
    shutil.copy('inputs/corpus/DaGuanVocabForElmo.txt','ELMo/200dim/DaGuanVocabForElmo.txt')
    shutil.copy('inputs/corpus/DaGuanVocabForElmo.txt','ELMo/250dim/DaGuanVocabForElmo.txt')
    shutil.copy('inputs/corpus/DaGuanVocabForElmo.txt','ELMo/300dim/DaGuanVocabForElmo.txt')

def preTrainELMo(dimNum,savePath):
    vocabPath='inputs/corpus/DaGuanVocabForElmo.txt'
    trainPrefix='inputs/corpus/processed_corpus.txt'
    train_elmo.preTrain(dimNum,vocabPath,savePath,trainPrefix,trainCharsNum=93245582)
            
def tokenEmbeddingFileToEmbedding(hdf5Path,savePath):
    fr=h5py.File(hdf5Path,'r')
    fr1=open(savePath,'w',encoding='utf-8')
    fr2 = open('inputs/corpus/DaGuan/DaGuanVocabForElmo.txt', 'r', encoding='utf-8')
    tokenList = fr2.readlines()
    for i in range(len(fr['embedding'])):
        embedding=fr['embedding'][i].tolist()
        embedding=list(map(str,embedding))
        token = tokenList[i].strip()
        fr1.write(token + ' ')
        fr1.write(' '.join(embedding)+'\n')

#——————————————————————————————————————构建所需语料———————————————————————————————————————
constructCorpusForDaGuan()
ConstructCorpusForELMo()

#——————————————————————————————————————150dim word2Vec,fastText预训练———————————————————————————————————————
getPreTrain(mode='char',dimNum=150,minCount=5,
            saveFileWord2Vec='inputs/embedding_matrix/150dim/DaGuanWord2Vec_150dim.txt',
            saveFileFastText='inputs/embedding_matrix/150dim/DaGuanFastText_150dim.txt',
            saveFileGlove='inputs/embedding_matrix/150dim/DaGuanGlove_150dim.txt')
getPreTrain(mode='bichar',dimNum=150,minCount=5,
            saveFileWord2Vec='inputs/embedding_matrix/150dim/DaGuanBicharWord2Vec_150dim.txt',
            saveFileFastText='inputs/embedding_matrix/150dim/DaGuanBicharFastText_150dim.txt',
            saveFileGlove='inputs/embedding_matrix/150dim/DaGuanBicharGlove_150dim.txt')
            
preTrainELMo(dimNum=150,savePath='ELMo/150dim/checkpointForElmo_150dim')
dw('ELMo/150dim/checkpointForElmo_150dim','ELMo/150dim/weights.hdf5')
shutil.copy('ELMo/150dim/checkpointForElmo_150dim/options.json','ELMo/150dim/options.json')
tokenEmbeddingFileToEmbedding(hdf5Path='ELMo/150dim/DaGuanElmo_150dim.hdf5',
                              savePath='inputs/embedding_matrix/150dim/DaGuanElmoPretrainEmbedding_150dim.txt')


#——————————————————————————————————————200dim word2Vec,fastText预训练———————————————————————————————————————
getPreTrain(mode='char',dimNum=200,minCount=5,
            saveFileWord2Vec='inputs/embedding_matrix/200dim/DaGuanWord2Vec_200dim.txt',
            saveFileFastText='inputs/embedding_matrix/200dim/DaGuanFastText_200dim.txt',
            saveFileGlove='inputs/embedding_matrix/200dim/DaGuanGlove_200dim.txt')
getPreTrain(mode='bichar',dimNum=200,minCount=5,
            saveFielWord2Vec='inputs/embedding_matrix/200dim/DaGuanBicharWord2Vec_200dim.txt',
            saveFileFastText='inputs/embedding_matrix/200dim/DaGuanBicharFastText_200dim.txt',
            saveFileGlove='inputs/embedding_matrix/200dim/DaGuanBicharGlove_200dim.txt')

preTrainELMo(dimNum=200,savePath='ELMo/200dim/checkpointForElmo_200dim')
dw('ELMo/200dim/checkpointForElmo_200dim','ELMo/200dim/weights.hdf5')
shutil.copy('ELMo/200dim/checkpointForElmo_200dim/options.json','ELMo/200dim/options.json')
tokenEmbeddingFileToEmbedding(hdf5Path='ELMo/200dim/DaGuanElmo_200dim.hdf5',
                              savePath='inputs/embedding_matrix/200dim/DaGuanElmoPretrainEmbedding_200dim.txt')

#——————————————————————————————————————250dim word2Vec,fastText预训练———————————————————————————————————————
getPreTrain(mode='char',dimNum=250,minCount=5,
            saveFileWord2Vec='inputs/embedding_matrix/250dim/DaGuanWord2Vec_250dim.txt',
            saveFileFastText='inputs/embedding_matrix/250dim/DaGuanFastText_250dim.txt',
            saveFileGlove='inputs/embedding_matrix/250dim/DaGuanGlove_250dim.txt')
getPreTrain(mode='bichar',dimNum=250,minCount=5,
            saveFielWord2Vec='inputs/embedding_matrix/250dim/DaGuanBicharWord2Vec_250dim.txt',
            saveFileFastText='inputs/embedding_matrix/250dim/DaGuanBicharFastText_250dim.txt',
            saveFileGlove='inputs/embedding_matrix/250dim/DaGuanBicharGlove_250dim.txt')


preTrainELMo(dimNum=250,savePath='ELMo/250dim/checkpointForElmo_250dim')
dw('ELMo/250dim/checkpointForElmo_250dim','ELMo/250dim/weights.hdf5')
shutil.copy('ELMo/250dim/checkpointForElmo_250dim/options.json','ELMo/250dim/options.json')
tokenEmbeddingFileToEmbedding(hdf5Path='ELMo/250dim/DaGuanElmo_250dim.hdf5',
                              savePath='inputs/embedding_matrix/250dim/DaGuanElmoPretrainEmbedding_250dim.txt')

#——————————————————————————————————————300dim word2Vec,fastText预训练———————————————————————————————————————
getPreTrain(mode='char',dimNum=300,minCount=5,
            saveFileWord2Vec='inputs/embedding_matrix/300dim/DaGuanWord2Vec_300dim.txt',
            saveFileFastText='inputs/embedding_matrix/300dim/DaGuanFastText_300dim.txt',
            saveFileGlove='inputs/embedding_matrix/300dim/DaGuanGlove_300dim.txt')
getPreTrain(mode='bichar',dimNum=300,minCount=5,
            saveFielWord2Vec='inputs/embedding_matrix/300dim/DaGuanBicharWord2Vec_300dim.txt',
            saveFileFastText='inputs/embedding_matrix/300dim/DaGuanBicharFastText_300dim.txt',
            saveFileGlove='inputs/embedding_matrix/300dim/DaGuanBicharGlove_300dim.txt')

preTrainELMo(dimNum=300,savePath='ELMo/300dim/checkpointForElmo_300dim')
dw('ELMo/300dim/checkpointForElmo_300dim','ELMo/300dim/weights.hdf5')
shutil.copy('ELMo/300dim/checkpointForElmo_300dim/options.json','ELMo/300dim/options.json')
tokenEmbeddingFileToEmbedding(hdf5Path='ELMo/300dim/DaGuanElmo_300dim.hdf5',
                              savePath='inputs/embedding_matrix/300dim/DaGuanElmoPretrainEmbedding_300dim.txt')


