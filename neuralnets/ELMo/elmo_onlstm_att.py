import logging
from bilm import  BidirectionalLanguageModel, weight_layers
from keras.optimizers import *
from keras.models import Model
from keras.layers import *

from neuralnets.keraslayers.ChainCRF import ChainCRF
from keras_self_attention import SeqSelfAttention
from keras_ordered_neurons import ONLSTM

class   ONLSTM_att:
    def __init__(self, char_word2vec, char_glove, char_fasttext,bichar_word2vec,bichar_glove,bichar_fasttext,
                 elmo_dim,params=None):
        # modelSavePath = Path for storing models, resultsSavePath = Path for storing output labels while training
        self.char_word2vec = char_word2vec
        self.char_glove = char_glove
        self.char_fasttext = char_fasttext

        self.bichar_word2vec = bichar_word2vec
        self.bichar_glove = bichar_glove
        self.bichar_fasttext = bichar_fasttext

        self.elmo_dim = elmo_dim
        # Hyperparameters for the network
        defaultParams = {'dropout': 0.25, 'LSTM-Size': (300,300),
                         'optimizer': 'adam', 'clipvalue': 0, 'clipnorm': 1, 'n_class_labels': 7}
        if params != None:
            defaultParams.update(params)
        self.params = defaultParams

    def build_model(self):


        char_input = Input(shape=(None,), dtype='int32', name='char_input')
        bichar_input = Input(shape=(None,),dtype='int32',name='bichar_input')
        elmo_input = Input(shape=(None,),dtype='int32',name='elmo_input')

        inputNodes = [char_input,bichar_input,elmo_input]

        word2vec_char_embedding = Embedding(input_dim=self.params['char2id_size'] + 1, output_dim=self.params['char_embedding_size']
                                   , trainable=False,weights=[self.char_word2vec], name='word2vec_char_embedding')(char_input)

        glove_char_embedding = Embedding(input_dim=self.params['char2id_size'] + 1, output_dim=self.params['char_embedding_size']
                                   , trainable=False,weights=[self.char_glove], name='glove_char_embedding')(char_input)

        fasttext_char_embedding = Embedding(input_dim=self.params['char2id_size'] + 1, output_dim=self.params['char_embedding_size']
                                   , trainable=False,weights=[self.char_fasttext], name='fasttext_char_embedding')(char_input)

        bichar_word2vec_embeding = Embedding(input_dim=self.params['bichar2id_size'] + 1, output_dim=self.params['bichar_embedding_size']
                                   , weights=[self.bichar_word2vec],trainable=False, name='word2vec_bichar_embedding')(bichar_input)

        bichar_glove_embedding = Embedding(input_dim=self.params['bichar2id_size'] + 1, output_dim=self.params['bichar_embedding_size']
                                   , weights=[self.bichar_glove],trainable=False, name='glove_bichar_embedding')(bichar_input)

        bichar_fasttext_embedding = Embedding(input_dim=self.params['bichar2id_size'] + 1, output_dim=self.params['bichar_embedding_size']
                                   , weights=[self.bichar_fasttext],trainable=False, name='fasttext_bichar_embedding')(bichar_input)


        word2vec_char_embedding  = Dropout(self.params['dropout'])(word2vec_char_embedding)
        glove_char_embedding = Dropout(self.params['dropout'])(glove_char_embedding)
        fasttext_char_embedding = Dropout(self.params['dropout'])(fasttext_char_embedding)

        bichar_word2vec_embeding = Dropout(self.params['dropout'])(bichar_word2vec_embeding)
        bichar_glove_embedding = Dropout(self.params['dropout'])(bichar_glove_embedding)
        bichar_fasttext_embedding = Dropout(self.params['dropout'])(bichar_fasttext_embedding)

        shared_layer = Concatenate(axis=-1)([word2vec_char_embedding,glove_char_embedding,fasttext_char_embedding,
                                             bichar_word2vec_embeding,bichar_glove_embedding,bichar_fasttext_embedding
                                           ])
        elmo_embedding = ELMoEmbedding(output_dim=self.elmo_dim*2,elmo_dim = self.elmo_dim)(elmo_input)

        shared_layer = Concatenate(axis=-1)([shared_layer,elmo_embedding])

        for size in self.params['LSTM-Size']:
            shared_layer = Bidirectional(ONLSTM(size,chunk_size = 30 ,return_sequences=True))(shared_layer)
            shared_layer = Dropout(self.params['dropout'])(shared_layer)

        self_att = SeqSelfAttention()(shared_layer)
        lstm_att = Concatenate(axis=-1)([shared_layer, self_att])

        output = lstm_att
        output = TimeDistributed(Dense(self.params['n_class_labels'], activation=None))(output)

        crf = ChainCRF()
        output = crf(output)
        lossFct = crf.sparse_loss

        # :: Parameters for the optimizer ::
        optimizerParams = {}
        if 'clipnorm' in self.params and self.params['clipnorm'] != None and self.params['clipnorm'] > 0:
            optimizerParams['clipnorm'] = self.params['clipnorm']

        opt = Adam(**optimizerParams)

        model = Model(inputs=inputNodes, outputs=[output])
        model.compile(loss=lossFct, optimizer=opt)

        model.summary(line_length=125)

        return model

class ELMoEmbedding(Layer):
    def __init__(self,output_dim , elmo_dim,**kwargs):
        super(ELMoEmbedding, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.elmo_dim = elmo_dim

    def build(self, input_shape):

        self.bilm = self.get_bilm()
        super(ELMoEmbedding, self).build(input_shape)

    def call(self, x,mask=None):
        context_embeddings_op = self.bilm(x)
        elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.001)
        elmo = elmo_context_input['weighted_op']

        return elmo

    def get_bilm(self):
        token_embedding_file = './ELMo/{}dim/DaGuanElmo_{}dim.hdf5'.format(self.elmo_dim,self.elmo_dim)
        options_file = './ELMo/{}dim/options.json'.format(self.elmo_dim)
        weight_file = './ELMo/{}dim/weights.hdf5'.format(self.elmo_dim)
        bilm = BidirectionalLanguageModel(
            options_file,
            weight_file,
            use_character_inputs=False,
            embedding_weight_file=token_embedding_file
        )

        return bilm

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1],self.output_dim)