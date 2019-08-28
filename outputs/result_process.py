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
    temp = json.load(open('./ELMo/bigger_size/200dim/bilstm_selfatt0.json', encoding='utf-8'))
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

    _load_result('./version_5/ELMo/200dim/',result,0.2226813024232553 )
    _load_result('./version_5/ELMo/250dim/', result,  0.18822566940797605)
    _load_result('./version_5/ELMo/300dim/',result,0.16813069311119425 )
    _load_result('./version_5/ELMo/200_250dim/',result,0.2330939548892124)
    _load_result('./version_5/ELMo/150_300dim/',result,0.18786838016836185)

    # _load_result('./version_4/word2vec_glove/200dim/', result, 0.15)
    # _load_result('./version_4/word2vec_glove/300dim/', result, 0.1)
    # _load_result('./version_4/word2vec_glove/150_300dim/', result, 0.1)
    # _load_result('./version_4/word2vec_glove/200_250dim/', result, 0.15)
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


def _safe_fix(label, ds_start_index, ds_end_index, start_index, end_index):
    for idx, _label in enumerate(label):
        if idx >= ds_start_index and idx <= ds_end_index:
            if idx >= start_index and idx <= end_index:
                continue
            else:
                if _label != 'O':
                    return False

    return True

def _fix_label(label, text, ds_entity, start_index, end_index, type):
    # 对label ,进行远程监督修复
    new_label = []
    # step 1 , 找到 ds_entity 在text中的所有，start_index,end_index
    #这里有BUG
    ds_positions = _find_start_index_end_index(text, ds_entity)
    # step 2 , 根据找到的下标，确定是否对我要修复的实体进行修复 （ds_entity的下标范围包含了原本的下标范围）
    # 查看寻找出来的下标，哪些符合修改
    # 1, 哪个start_index 和 end_index在哪个实体范围内
    flag = 0
    for position in ds_positions:
        if start_index >= position[0] and end_index <= position[1]:
            ds_start_index = position[0]
            ds_end_index = position[1]
            flag = 1
            break
    if flag == 0:
        return label
    # 2 判断在要修改的范围是不是安全的
    #这里有问题
    if _safe_fix(label, ds_start_index, ds_end_index, start_index, end_index):
        for idx, _label in enumerate(label):
            if idx == ds_start_index:
                new_label.append('B-' + type)
            elif idx > ds_start_index and idx <= ds_end_index:
                new_label.append('I-' + type)
            else:
                new_label.append(_label)
        return new_label
    else:
        return label


def fix_result(result,final_label):
    #按照远程监督的方法对bio的边界问题进行修复。
    """
    result:dic{'text','pred_bio'}
    :param result:
    :param final_label:
    :return:
    """
    a_entities, b_entities, c_entities, a_entities_text, b_entities_text, c_entities_text = _get_entities_text_type(_load_train_data())
    #先查看一下有没有分错类的
    entities = []
    posistions = []
    count = 0
    count2 = 0
    for idx,data in enumerate(result[0]):
        _entities,_posistions = (_get_entity(data['text'],final_label[idx]))
        entities.append(_entities)
        posistions.append(_posistions)

    #1，解决没有截取完全的问题

    for idx,entity in enumerate(entities):

        for _idx, _entity in enumerate(entity):

            entity_text = _entity[0]
            entity_type = _entity[1]

            if entity_type == 'a' and entity_text not in a_entities_text:

                for ds_entity in a_entities_text:
                    if entity_text in ds_entity and entity_text != ds_entity and ds_entity in '_'.join(result[0][idx]['text']): #有可能出现没有截取完全的情况
                        count2 +=1
                        start_index = posistions[idx][_idx][0]
                        end_index = posistions[idx][_idx][1]
                        _final_label = _fix_label(final_label[idx],result[0][idx]['text'],ds_entity,start_index,end_index,'a')
                        if _final_label != final_label[idx]:
                            final_label[idx] = _final_label
                            count+=1

            if entity_type == 'b' and entity_text not in b_entities_text:

                for ds_entity in b_entities_text:
                    if entity_text in ds_entity and entity_text != ds_entity and ds_entity in '_'.join(result[0][idx]['text']): #有可能出现没有截取完全的情况
                        count2 +=1

                        start_index = posistions[idx][_idx][0]
                        end_index = posistions[idx][_idx][1]
                        _final_label = _fix_label(final_label[idx],result[0][idx]['text'],ds_entity,start_index,end_index,'a')
                        if _final_label != final_label[idx]:
                            final_label[idx] = _final_label
                            count+=1


            if entity_type == 'c' and entity_text not in c_entities_text:

                for ds_entity in b_entities_text:
                    if entity_text in ds_entity and entity_text != ds_entity and ds_entity in '_'.join(result[0][idx]['text']):  # 有可能出现没有截取完全的情况
                        count2 +=1
                        start_index = posistions[idx][_idx][0]
                        end_index = posistions[idx][_idx][1]
                        _final_label = _fix_label(final_label[idx],result[0][idx]['text'],ds_entity,start_index,end_index,'a')
                        if _final_label != final_label[idx]:
                            final_label[idx] = _final_label
                            count+=1

    print('远程监督修复边缘问题实体{}条'.format(count))
    print('触发{}条'.format(count2))

def _get_free_text(text, label):
    # 将文本中没有在BIO中出现的文本提取出来
    free_text = []
    temp = []
    for idx, char in enumerate(text):

        if label[idx] == 'O':
            temp.append(char)
        elif temp:
            free_text.append('_'.join(temp))
            temp = []
    if temp:
        free_text.append('_'.join(temp))
    return free_text

def _add_ds_label(entity,text,label,type):
    #找出text，entity，标注成BIO

    positions = _find_start_index_end_index(text,entity)
    _flag = 0
    for position in positions:
        start_index = position[0]
        end_index = position[1]
        flag = 1
        for idx,_label in enumerate(label):
            if idx >= start_index and idx <= end_index and _label != 'O':
                flag = 0
                _flag = 1
                break
        if flag == 1 :
            break

    #找到能改的下标
    if _flag == 1:
        #将对应下标标为BIO
        label[start_index] = 'B-'+ type

        for i in range(start_index+1,end_index+1):
            label[i] = 'I-' + type

    return label

def _add_ds_entity(ds_entities,text,label):
    #约束条件，出现大于等于5，
    # 取每个可能的ds_entity中最长的
    """
    ds_entities. [[list1].[]] list1保存了这个段中所有有可能的ds_entity
    :param ds_entities:
    :param label:
    :return:
    """
    temp = [] #保存要进行远程监督实体和他的属性
    ds_entity = []
    for entities in ds_entities:
        #找出长度长的那个
        if entities:
            #找出最长的那个实体
            entities_text = [entity['entity'] for entity in entities]
            entitiy_text = max(entities_text, key=len)
            #找出最长那个实体的字典
            for entity in entities:
                if entity['entity'] == entitiy_text:
                    temp.append(entity)
    ds_entity.extend(temp)

    for entity in ds_entity:
        if entity['property']['a_num'] >=5 or entity['property']['b_num'] >=5  or entity['property']['c_num'] >=5:
            #选出最大的。
            if  entity['property']['a_num'] >  entity['property']['b_num'] and entity['property']['a_num'] > entity['property']['c_num']:

                label = _add_ds_label(entity['entity'],text,label,'a')

            elif  entity['property']['b_num'] >  entity['property']['a_num'] and entity['property']['b_num'] > entity['property']['c_num']:
                label = _add_ds_label(entity['entity'],text,label,'b')

            if  entity['property']['c_num'] >  entity['property']['b_num'] and entity['property']['c_num'] > entity['property']['a_num']:
                label = _add_ds_label(entity['entity'],text,label,'c')

    return label


#todo , 看一下远程监督提高召回的可能性,是否存在，实体百分百出现。
def ds_entity(result,final_label):
    #约束条件， 单字的就不弄了， 另外出现次数大于（这个等统计做好用百分比代替）
    #远程监督补召回,如果出现子串情况，用最大匹配串进行远程监督
    count = 0
    entities = _load_train_data()
    a_entities, b_entities, c_entities, \
    a_entities_text, b_entities_text, c_entities_text = _get_entities_text_type(entities)

    #将实体做成字典
    ds_entities = {}
    for entity in a_entities_text:
        if entity not in ds_entities:
            ds_entities[entity] = {}
            ds_entities[entity]['a_num'] = 1
            ds_entities[entity]['b_num'] = 0
            ds_entities[entity]['c_num'] = 0

        else:
            ds_entities[entity]['a_num'] += 1

    for entity in b_entities_text:
        if entity not in ds_entities:
            ds_entities[entity] = {}
            ds_entities[entity]['a_num'] = 0
            ds_entities[entity]['b_num'] = 1
            ds_entities[entity]['c_num'] = 0
        else:
            ds_entities[entity]['b_num'] += 1

    for entity in c_entities_text:
        if entity not in ds_entities:
            ds_entities[entity] = {}
            ds_entities[entity]['a_num'] = 0
            ds_entities[entity]['b_num'] = 0
            ds_entities[entity]['c_num'] = 1
        else:
            ds_entities[entity]['c_num'] += 1

    #step1, 找出文本中没有构成实体的文本段
    free_text = []

    for idx,data in enumerate(result[0]):
        free_text.append(_get_free_text(data['text'],final_label[idx]))

    #step2, 遍历每个样本的free_text， 进行远程监督.将可能的实体保存起来
    ds_set = []
    for idx,_free_text in tqdm(enumerate(free_text)):
        temp = [] #保存一句话所有text段的可能是提
        for _idx,text in enumerate(_free_text):
            _temp = [] #保存某段free_text所有可能实体
            for ds_entity,property in ds_entities.items():
                if ds_entity in text and len(ds_entity.split('_'))>1:
                    dic ={}
                    dic['entity'] = ds_entity
                    dic['property'] = property
                    _temp.append(dic)
            temp.append(_temp)
        ds_set.append(temp)

    for idx,label in enumerate(final_label):
        final_label[idx]  = _add_ds_entity(ds_set[idx],result[0][idx]['text'],label)

if __name__ == '__main__':

    result = load_result()
    final_label = vote_result(result)

    #找出哪些投票样本出现了错误BIO

    final_label = clean_voted_label(final_label)

    # #用远程监督进行修复
    # fix_result_type(result, final_label)
    # # fix_result(result, final_label)
    # # ds_entity(result,final_label)
    #
    submmition = submit_format(final_label,result)
    output_submmition(submmition)