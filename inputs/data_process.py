import json
import codecs
import numpy as np
import random

def get_bichar(text):
    """
    w1,w2,w3,w4,w5,w6,w7
    bichar : w1w2,w2w3,w4w5,w5w6,w7w0
    :param text:
    :return:
    """
    new_text = []
    for index, char in enumerate(text):
        if index != len(text) - 1:  # 没有达到最后一个
            new_text.append('_'.join([char,text[index + 1]]))
        else:
            new_text.append('_'.join([char, '$']))
    return new_text

def load_train_data():
    train_data = []
    entities = {'a':[],'b':[],'c':[]}
    with open('./train.txt','r',encoding='utf-8') as fr:
        for line in fr:
            dic = {}
            sentence = line.strip().split('  ')
            label = []
            text = []
            for word_entity in sentence:
                words = word_entity.split('/')[0].split('_')
                entity_type = word_entity.split('/')[1]

                if entity_type == 'o':
                    label += ['O']*len(words)
                else:
                    label += ['B-' + entity_type]
                    label += ['I-' + entity_type] * (len(words) -1)

                    if entity_type == 'a':
                        if words not in entities['a']:
                            entities['a'].append(words)
                    elif entity_type == 'b':
                        if words not in entities['b']:
                            entities['b'].append(words)
                    elif entity_type == 'c':
                        if words not in entities['c']:
                            entities['c'].append(words)

                text += words

            dic['bichar'] = get_bichar(text)
            dic['text'] = text
            dic['bio'] = label
            train_data.append(dic)

    return train_data,entities


def collect_entities(index):
    entities = {'a':[],'b':[],'c':[]}
    _index = 0
    with open('./inputs/train.txt','r',encoding='utf-8') as fr:
        for line in fr:
            if _index in index:
                sentence = line.strip().split('  ')
                for word_entity in sentence:
                    words = word_entity.split('/')[0].split('_')
                    entity_type = word_entity.split('/')[1]

                    if entity_type == 'o':
                        continue
                    else:
                        if entity_type == 'a':
                            if words not in entities['a']:
                                entities['a'].append(words)
                        elif entity_type == 'b':
                            if words not in entities['b']:
                                entities['b'].append(words)
                        elif entity_type == 'c':
                            if words not in entities['c']:
                                entities['c'].append(words)
            _index+=1
    return entities

#每个样本产生5个数据增强数据
def load_AG_data(entities):
    AG_data = []
    with open('./inputs/train.txt','r',encoding='utf-8') as fr:
        for line in fr:
            dic = {}
            sentence = line.strip().split('  ')
            label = []
            text = []
            for word_entity in sentence:
                words = word_entity.split('/')[0].split('_')
                entity_type = word_entity.split('/')[1]

                if entity_type == 'o':
                    label += ['O']*len(words)
                else:
                    while True:
                        _words = np.random.choice(entities[entity_type])
                        if _words != words:
                            words = _words
                            break
                    label += ['B-' + entity_type]
                    label += ['I-' + entity_type] * (len(words) -1)

                text += words

            dic['bichar'] = get_bichar(text)
            dic['text'] = text
            dic['bio'] = label
            AG_data.append(dic)
    return AG_data


def load_test_data():
    test_data = []
    with open('./test.txt','r',encoding='utf-8') as fr:
        for line in fr:
            dic = {}
            dic['text'] = line.strip().split('_')
            dic['bichar'] = get_bichar(dic['text'])
            # dic['trichar'] = get_trichar(dic['text'])
            test_data.append(dic)
    return test_data


def collect_char2id(datasets, save_file='char2id.json'):
    chars = {}
    for data in datasets:
        for char in data['text']:
                chars[char] = chars.get(char, 0) + 1
    chars = {i:j for i,j in chars.items()  }
    id2char = {i + 1: j for i, j in enumerate(chars)}  # padding: 0
    char2id = {j: i for i, j in id2char.items()}
    with codecs.open(save_file, 'w', encoding='utf-8') as f:
        json.dump([id2char, char2id], f, indent=4, ensure_ascii=False)

def collect_label2id(datasets, save_file='bio2id.json'):
    labels = {}
    for data in datasets:
        for label in data['bio']:
            if label != 'O':
                labels[label] = labels.get(label, 0) + 1
    labels = {i:j for i,j in labels.items() }
    id2label = {i + 1: j for i, j in enumerate(labels)}  # UNK0
    id2label[0] = 'O'
    label2id = {j: i for i, j in id2label.items()}
    with codecs.open(save_file, 'w', encoding='utf-8') as f:
        json.dump([id2label, label2id], f, indent=4, ensure_ascii=False)

def collect_bichar2id(datasets, save_file='bichar2id.json'):
    bichars = {}
    for data in datasets:
        for bichar in data['bichar']:
                bichars[bichar] = bichars.get(bichar, 0) + 1
    bichars = {i:j for i,j in bichars.items() }
    id2bichar = {i + 1: j for i, j in enumerate(bichars)}  # UNK0
    bichar2id = {j: i for i, j in id2bichar.items()}
    with codecs.open(save_file, 'w', encoding='utf-8') as f:
        json.dump([id2bichar, bichar2id], f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    train_data,entities  = load_train_data()
    AG_data = load_AG_data(entities)

    test_data = load_test_data()
    collect_char2id(train_data+test_data)
    collect_label2id(train_data)
    collect_bichar2id(train_data+test_data)

    with codecs.open('./train_data.json','w',encoding='utf-8') as fr:
        json.dump(train_data,fr,indent=4, ensure_ascii=False)

    with codecs.open('./AG_data.json','w',encoding='utf-8') as fr:
         json.dump(AG_data,fr,indent=4, ensure_ascii=False)

    with codecs.open('./test_data.json','w',encoding='utf-8') as fr:
        json.dump(test_data,fr,indent=4, ensure_ascii=False)