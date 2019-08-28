import keras
from keras import backend as K
from keras.layers import *


def seq_and_vec(x):
    """seq是[None, seq_len, s_size]的格式，
    vec是[None, v_size]的格式，将vec重复seq_len次，拼到seq上，
    得到[None, seq_len, s_size+v_size]的向量。
    """
    seq, vec = x
    vec = K.expand_dims(vec, 1)
    vec = K.zeros_like(seq[:, :, :1]) + vec
    return K.concatenate([seq, vec], 2)

def attention_like_tensor(x):
    """
    把attention向量广播到每个一时间步，
    :param x: [batch,dim]
    :return:  [batch,sentene,dim]
    """

class Attention_Layer(keras.layers.Layer):

    """
    dot attention for word_char_embedding
    q,v,k for define the attention
    score = softmax(dot(q,v))
    attention = sum(score*k)=
    """
    # def __init__(self,**kwargs):
    #     super(Attention_Layer,self).__init__(**kwargs)
    #
    # def build(self, input_shape):
    #     self.W = self.add_weight(name='W',shape=(input_shape[-1],input_shape[-1]),initializer='glorot_normal')
    #     self.acitvation =
    # def call(self,inputs,mask=None):
    #     score = K.softmax(K.dot(inputs,self.W),axis=-1)
    #     c =


class Gate_Add_Lyaer(keras.layers.Layer):
    """
    gate add mechanism for word_char embedding
    z =  sigmoid(W(1)tanh(W(2)word_embedding + W(3)char_att))
    word_char_embedding = z*word_embedding + (1-z)char_att
    """
    def __init__(self,**kwargs):
        """

        :param word_embedding:  shape [batch,sentence,dim of word_embedding]
        :param char_att:  shape [batch,sentence,dim of char_embedding]
        :param kwargs:
        """
        super(Gate_Add_Lyaer,self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        assert input_shape[0][2] == input_shape[1][2]

        self.W1 = self.add_weight(name='W1',shape=(input_shape[0][-1],input_shape[0][-1]),initializer='glorot_normal') #[dim,dim]
        self.W2 = self.add_weight(name='W2',shape=(input_shape[0][-1],input_shape[0][-1]),initializer='glorot_normal')
        self.W3 = self.add_weight(name='W3',shape=(input_shape[0][-1],input_shape[0][-1]),initializer='glorot_normal')

        super(Gate_Add_Lyaer, self).build(input_shape)

    def call(self,inputs,mask=None):
        # inputs[0]:word_embedding ,inputs[1]:char_embedding
        word_embedding_shape = K.int_shape(inputs[0]) #[batch,sentence,dim of word embedding]
        char_embedding_shape = K.int_shape(inputs[1]) #[batch,sentence,dim of char embedding]
        # word_embedding_reshaped = K.reshape(inputs[0],shape=(-1,word_embedding_shape[-1])) #[batch*sentence,dim of word embedding]
        # char_embedding_reshaped = K.reshape(inputs[1],shape=(-1,char_embedding_shape[-1])) #[batch*sentence, dim of char embedding]
        word_embedding = K.dot(inputs[0],self.W1)
        char_embedding = K.dot(inputs[1],self.W2)
        wc_tanh = K.tanh(word_embedding+char_embedding)
        z = K.sigmoid(K.dot(wc_tanh,self.W3))
        embedding = z*inputs[0]+(1-z)*inputs[1]
        # z = K.sigmoid(K.dot(K.tanh(K.dot(word_embedding_reshaped,self.W1) + K.dot(char_embedding_shape,self.W2)),self.W3))
        # embedding = z*word_embedding_reshaped + (1-z)*char_embedding_reshaped  #[batch*sentence,]
        # embedding = K.reshape(embedding,shape=(-1,word_embedding_reshaped[1],word_embedding_reshaped[-1]))# [batch,sentecen,dim]
        return embedding

    def compute_mask(self, inputs, mask=None):
        return mask


    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],input_shape[0][1],input_shape[0][2])


class Position_Embedding(Layer):

    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size  # 必须为偶数
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x,mask=None):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
        position_j = 1. / K.pow(10000., 2 * K.arange(self.size / 2, dtype='float32') / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1  # K.arange不支持变长，只好用这种方法生成
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)


class Self_Attention_Layer(Layer):
    """多头注意力机制
       """

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.out_dim = nb_head * size_per_head
        super(Self_Attention_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        q_in_dim = input_shape[0][-1]
        k_in_dim = input_shape[1][-1]
        v_in_dim = input_shape[2][-1]
        self.q_kernel = self.add_weight(name='q_kernel',
                                        shape=(q_in_dim, self.out_dim),
                                        initializer='glorot_normal')
        self.k_kernel = self.add_weight(name='k_kernel',
                                        shape=(k_in_dim, self.out_dim),
                                        initializer='glorot_normal')
        self.v_kernel = self.add_weight(name='w_kernel',
                                        shape=(v_in_dim, self.out_dim),
                                        initializer='glorot_normal')

    def mask(self, x, mask, mode='mul'):
        if mask is None:
            return x
        else:
            for _ in range(K.ndim(x) - K.ndim(mask)):
                mask = K.expand_dims(mask, K.ndim(mask))
            if mode == 'mul':
                return x * mask
            else:
                return x - (1 - mask) * 1e10

    def call(self, inputs):
        q, k, v = inputs[:3]
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变化
        qw = K.dot(q, self.q_kernel)
        kw = K.dot(k, self.k_kernel)
        vw = K.dot(v, self.v_kernel)
        # qw = Dense(self.out_dim,activation='relu')(q)
        # kw = Dense(self.out_dim, activation='relu')(k)
        # vw = Dense(self.out_dim, activation='relu')(v)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.nb_head, self.size_per_head))
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.nb_head, self.size_per_head))
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.nb_head, self.size_per_head))
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 2, 1, 3))
        kw = K.permute_dimensions(kw, (0, 2, 1, 3))
        vw = K.permute_dimensions(vw, (0, 2, 1, 3))
        # Attention
        a = K.batch_dot(qw, kw, [3, 3]) / self.size_per_head ** 0.5
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = self.mask(a, v_mask, 'add')
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = K.softmax(a)
        # 完成输出
        o = K.batch_dot(a, vw, [3, 2])
        o = K.permute_dimensions(o, (0, 2, 1, 3))
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = self.mask(o, q_mask, 'mul')
        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)


class MaskedConv1D(keras.layers.Conv1D):

    def __init__(self, **kwargs):
        super(MaskedConv1D, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            inputs *= K.expand_dims(mask, axis=-1)
        return super(MaskedConv1D, self).call(inputs)


class MaskedLSTM(keras.layers.CuDNNLSTM):

    def __init__(self, **kwargs):
        super(MaskedLSTM, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, mask=None, training=None, initial_state=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            inputs *= K.expand_dims(mask, axis=-1)
        return super(MaskedLSTM, self).call(inputs)


class MaskFlatten(keras.layers.Flatten):

    def __init__(self, **kwargs):
        super(MaskFlatten, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, mask=None):
        # if mask is not None:
            # mask = K.cast(mask, K.floatx())
            # inputs *= K.expand_dims(mask, axis=-1)
        return super(MaskFlatten, self).call(inputs) #调用父类的call ,然后传入inputs


class MaskRepeatVector(keras.layers.RepeatVector):

    def __init__(self, n,**kwargs):
        super(MaskRepeatVector, self).__init__(n,**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, mask=None):
        return super(MaskRepeatVector, self).call(inputs)

class MaskPermute(keras.layers.Permute):

    def __init__(self, dims,**kwargs):
        super(MaskPermute, self).__init__(dims,**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, mask=None):
        return super(MaskPermute, self).call(inputs)

class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable=True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                      as_dict=True,
                      signature='default',
                      )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)
