#每个token，取出现次数最多的
import json
from collections import Counter
import ipdb
from tqdm import tqdm
model1 = 'bilstm_cnn'
model2 = 'bilstm_selfatt'
model3 = 'local_att_gru'
models = [model1,model2,model3]

#目前最好全权重
self_att_weights = 0.540007597906051
bilstm_cnn_weights = 0.3077485980761171
hiera_gru_weights = 0.15224380401783202


def _load_result(result_path,result,result_weights):

    #result  [[{a:num,b:num},{}],[]]
    for model_name in models:

        if model_name == 'bilstm_selfatt':
            model_weights = self_att_weights
        elif  model_name == 'bilstm_cnn':
            model_weights = bilstm_cnn_weights
        elif model_name == 'local_att_gru':
            model_weights = hiera_gru_weights

        for i in range(5):
            path = result_path+str(model_name)+str(i)+'.json'
            temp = json.load(open(path,encoding='utf-8'))

            for idx,labels in enumerate(temp):

                for _idx,label in enumerate(labels['pred_bio']):
                    voted_labels = result[idx]['bio'][_idx]
                    if label in voted_labels:
                        result[idx]['bio'][_idx][label] +=  result_weights * model_weights
                    else:
                        result[idx]['bio'][_idx][label] =  result_weights * model_weights

def load_result():

    result = []
    #对result进行初始化
    temp = json.load(open('./200dim/bilstm_selfatt0.json', encoding='utf-8'))
    for idx,data in enumerate(temp):
        dic = {}
        dic['text'] = data['text']
        _temp = []
        for _idx,_ in enumerate(data['text']):
            _dic = {}
            _temp.append(_dic)
        dic['bio'] = _temp
        result.append(dic)

    #0.9349

    _load_result('./200dim/',result,0.2226813024232553 )
    _load_result('./250dim/', result,  0.18822566940797605)
    _load_result('./300dim/',result,0.16813069311119425 )
    _load_result('./200_250dim/',result,0.2330939548892124)
    _load_result('./150_300dim/',result,0.18786838016836185)

    return result


def vote_result(result):
    final_label = []
    for idx,data in enumerate(result):
        temp = []
        labels = data['bio']
        for _idx,label in enumerate(labels):
            sorted_label = sorted(label.items(), key=lambda item: item[1])
            temp.append(sorted_label[-1][0])
        final_label.append(temp)
    return final_label

def _clean_label(label):
    """
    :param label: [O', 'O', 'I-a', 'I-a', 'I-a', 'I-a', 'O', 'O', 'O', 'O', 'O']
    :return: [O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    """
    new_label = []
    B_flag = 0
    for _label in label:
        if _label =='O':
            new_label.append(_label)
            B_flag = 0
        else:
            bio_label = _label.split('-')[0]
            type = _label.split('-')[1]
            if bio_label == 'B':
                new_label.append(_label)
                B_flag = 'B'+type
            else: #I的情况， 判断B_flag是否符合
                if B_flag == 'B'+type:
                    new_label.append(_label)
                else:
                    new_label.append('O')
    return new_label

def clean_voted_label(final_label):
    # 对投票出来，不合理的情况进行清洗。
    #1, 对没有出现B ， 只有I出现的实体进行清除。 暂时使用最极端的情况
    fix_label =  []
    count = 0
    for idx,label in enumerate(final_label):
        _label = _clean_label(label)
        fix_label.append(_label)

        if _label != label:
            count+=1
    print('删除了{}个，BIO不符合的情况'.format(count))

    return fix_label

def find_wrong_voted_label_index(final_label):
    # 对投票出来，不合理的情况进行清洗。
    #1, 对没有出现B ， 只有I出现的实体进行清除。 暂时使用最极端的情况
    fix_label =  []
    count = 0

    fix_index=  []
    for idx,label in enumerate(final_label):
        _label = _clean_label(label)
        fix_label.append(_label)

        if _label != label:
            count+=1
            fix_index.append(idx)
    print('找到了{}个，BIO不符合的情况'.format(count))

    return fix_index

def submit_format(final_label,result):
    #将最终结果转换成提交格式
    submit_format = []
    for index in range(len(final_label)):
        text = result[index]['text']
        label = final_label[index]

        temp = []
        temp_submit = ''
        for idx,_label in enumerate(label):
            if _label == 'O':
                temp.append(text[idx])

                #达到最后一个O
                if idx+1 == len(label) or label[idx+1] != 'O':
                    temp_submit+= '_'.join(temp)
                    temp_submit += '/o  '
                    temp = []

            else:# 当label为a,b,c
                type = _label.split('-')[1]
                temp.append(text[idx])

                #已经对不合理的进行清晰了， 这样应该没问题
                if idx+1 == len(label) or label[idx+1].split('-')[0] == 'B' or label[idx+1] == 'O' or label[idx+1].split('-')[1] != type:
                    temp_submit+= '_'.join(temp)
                    temp_submit += '/' + str(type) + '  '
                    temp = []
        submit_format.append(temp_submit)
    return submit_format

def output_submmition(submmition,path='result.txt'):
    with open(path,'w',encoding='utf-8') as fr:
        for sub in submmition:
            fr.write(sub.strip('  ') + '\n')

def find_wrong_position(labels):
    last_label = 'start'
    start_index, end_index = -1, -1
    for idx, label in enumerate(labels):
        # 找到
        if last_label == 'O' and label.split('-')[0] == 'I':
            start_index = idx
            last_label = label

            for i in range(idx, len(labels)):
                if labels[i] != label:  # 找到end_index
                    end_index = i - 1
                    break
        else:
            last_label = label

    return (start_index, end_index)


def _load_train_data():
    entities = []
    with open('../inputs/train.txt', 'r', encoding='utf-8') as fr:
        for line in fr:
            sentence = line.strip().split('  ')
            for word_entity in sentence:
                words = word_entity.split('/')[0].split('_')
                entity_type = word_entity.split('/')[1]

                if entity_type == 'o':
                    pass
                else:
                    entities.append('_'.join(words) + '/' + entity_type)

    return entities

def _get_entities_text_type(entities):
    # 每个类型的实体做一个 文本字典
    a_entities_text = []
    a_entities = []
    b_entities_text = []
    b_entities = []
    c_entities_text = []
    c_entities = []
    for entity in entities:
        entity_type = entity.split('/')[1]
        entity_text = entity.split('/')[0]
        if entity_type == 'a' and entity_text not in a_entities:
            a_entities_text.append(entity_text)
            a_entities.append(entity)
        elif entity_type == 'b' and entity_text not in b_entities:
            b_entities_text.append(entity_text)
            b_entities.append(entity)
        elif entity_type == 'c' and entity_text not in c_entities:
            c_entities_text.append(entity_text)
            c_entities.append(entity)
    return a_entities, b_entities, c_entities,\
           a_entities_text, b_entities_text, c_entities_text

def _get_entity(text, labels):
    entity = []
    _entity = []
    position_ = []
    start_index = 0
    flag = 0
    for idx, label in enumerate(labels):

        if label != 'O':
            position = label.split('-')[0]
            type = label.split('-')[1]

            if position == 'B':
                start_index = idx
                _entity.append(text[idx])
                flag = 1
            elif position == 'I':
                _entity.append(text[idx])

        elif label == 'O':
            if _entity and flag == 1:
                end_index = idx
                entity.append(('_'.join(_entity), type))
                position_.append((start_index, end_index - 1))
                _entity = []
                flag = 0

        if idx == len(labels) - 1:

            if _entity and flag == 1:
                end_index = idx
                entity.append(('_'.join(_entity), type))
                position_.append((start_index, end_index))
                _entity = []
    return entity, position_


def _change_type(label,start_index,end_index,type):
    new_label = []
    for idx,_label in enumerate(label):
        #对范围内的label的类型进行修改
        if idx >= start_index and idx <= end_index:
            old_label = _label.split('-')
            old_label[1] = type
            new_label.append('-'.join(old_label))
        else:
            new_label.append(_label)
    return new_label

def fix_result_type(result,final_label):
    #按照远程监督的方法对bio的类别问题进行修复
    a_entities, b_entities, c_entities, a_entities_text, b_entities_text, c_entities_text = _get_entities_text_type(_load_train_data())
    entities = []
    posistions = []
    count = 0
    for idx,data in enumerate(result):
        _entities,_posistions = (_get_entity(data['text'],final_label[idx]))
        entities.append(_entities)
        posistions.append(_posistions)

    for idx,entity in enumerate(entities):

        for _idx, _entity in enumerate(entity):
            entity_text = _entity[0]
            entity_type = _entity[1]

            #a类实体在远程监督中，在,b,c类出现
            if entity_type == 'a'  and entity_text not in a_entities_text:
                if  entity_text in b_entities_text :
                    start_index = posistions[idx][_idx][0]
                    end_index = posistions[idx][_idx][1]
                    final_label[idx] = _change_type(final_label[idx],start_index,end_index,'b')
                    count+=1
                    break
                elif entity_text in c_entities_text:
                    start_index = posistions[idx][_idx][0]
                    end_index = posistions[idx][_idx][1]
                    final_label[idx] = _change_type(final_label[idx],start_index,end_index,'c')
                    count +=1
            elif entity_type == 'b' and entity_text not in b_entities_text:
                if entity_text in a_entities_text:
                    start_index = posistions[idx][_idx][0]
                    end_index = posistions[idx][_idx][1]
                    final_label[idx] = _change_type(final_label[idx],start_index,end_index,'a')
                    count+=1
                    break
                elif entity_text in c_entities_text:
                    start_index = posistions[idx][_idx][0]
                    end_index = posistions[idx][_idx][1]
                    final_label[idx] = _change_type(final_label[idx],start_index,end_index,'c')
                    count += 1
            elif entity_type == 'b' and entity_text not in b_entities_text:
                if entity_text in a_entities_text:
                    start_index = posistions[idx][_idx][0]
                    end_index = posistions[idx][_idx][1]
                    final_label[idx] = _change_type(final_label[idx],start_index,end_index,'a')
                    count += 1
                    break
                elif entity_text in c_entities_text:
                    start_index = posistions[idx][_idx][0]
                    end_index = posistions[idx][_idx][1]
                    final_label[idx] = _change_type(final_label[idx],start_index,end_index,'c')
                    count += 1

    print('根据远程监督一共修复{}条类别错误信息'.format(count))


def _find_start_index_end_index(text, entity):
    # 找到text中所有entity的下标范围
    _pos = 0
    position = []
    entity = entity.split('_')
    start_index = 0
    flag = 0  # 标记是否在寻找中
    for idx, char in enumerate(text):
        if char == entity[0] and flag == 0:
            start_index = idx
            flag = 1
            _pos += 1
        elif char == entity[_pos] and _pos != len(entity) - 1:
            _pos += 1

        elif _pos == len(entity) - 1:  # 完成匹配
            end_index = idx
            position.append((start_index, end_index))
            flag = 0
            _pos = 0
        else:
            flag = 0
            _pos = 0
    return position



if __name__ == '__main__':

    result = load_result()
    final_label = vote_result(result)

    #找出哪些投票样本出现了错误BIO

    final_label = clean_voted_label(final_label)
    #
    submmition = submit_format(final_label,result)
    output_submmition(submmition)