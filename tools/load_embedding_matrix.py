import numpy as np

def _load_embed(file,dim):
    def get_coefs(word, *arr):
        return word, np.asarray(arr)[:dim]
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='utf-8'))

    return embeddings_index

def _load_embedding_matrix(word_index, embedding,dim):
    embed_word_count = 0
    # nb_words = min(max_features, len(word_index))
    nb_words = len(word_index)
    embedding_matrix = np.random.normal(size=(nb_words+1, dim))

    for word, i in word_index.items():

#        if i >= max_features: continue
        if word not in embedding:
            word = word.lower()
        if word.islower and word not in embedding:
            word = word.title()
        embedding_vector = embedding.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            embed_word_count += 1
    print('词向量的覆盖率为{}'.format(embed_word_count / len(word_index)))
    return embedding_matrix

def _load_bichar_embedding_matrix(word_index, bichar_embedding,char_embedding,dim):
    embed_word_count = 0
    # nb_words = min(max_features, len(word_index))
    nb_words = len(word_index)
    embedding_matrix = np.random.normal(size=(nb_words+1, dim))

    for word, i in word_index.items():

        embedding_vector = bichar_embedding.get(word)

        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            embed_word_count += 1

        else:
            char1,char2 = word.split('_')
            char1_embedding_vector = char_embedding.get(char1)
            char2_embedding_vector = char_embedding.get(char2)
            if char1_embedding_vector is not None and char2_embedding_vector is not None:
                char1_embedding_vector = np.array(char1_embedding_vector, dtype='float32')
                char2_embedding_vector = np.array(char2_embedding_vector, dtype='float32')

                embedding_vector = np.mean([char1_embedding_vector, char2_embedding_vector], axis=0)
                embedding_matrix[i] = embedding_vector

    print('词向量的覆盖率为{}'.format(embed_word_count / len(word_index)))
    return embedding_matrix


def get_word2vec(word_index,path,dim):
    # embedding_dir = './inputs/embedding_matrix/150dim/DaGuanWord2Vec_150dim.txt'
    embedding = _load_embed(path,dim)
    embedding_matrix = _load_embedding_matrix(word_index, embedding,dim)

    return embedding_matrix

def get_glove(word_index,path,dim):
    embedding_dir = path
    embedding = _load_embed(embedding_dir,dim)
    embedding_matrix = _load_embedding_matrix(word_index, embedding,dim)

    return embedding_matrix


def get_fasttext(word_index,path,dim):
    # embedding_dir = './inputs/embedding_matrix/150dim/DaGuanFastText_150dim.txt'
    embedding = _load_embed(path,dim)
    embedding_matrix = _load_embedding_matrix(word_index, embedding,dim)

    return embedding_matrix


def get_bichar_word2vec(word_index,char_path,bichar_path,dim):

    char_embedding_dir = char_path
    bichar_embedding_dir = bichar_path
    char_embedding = _load_embed(char_embedding_dir,dim)
    bichar_embedding = _load_embed(bichar_embedding_dir,dim)

    embedding_matrix = _load_bichar_embedding_matrix(word_index, bichar_embedding,char_embedding,dim)

    return embedding_matrix


def get_bichar_fasttext(word_index,char_path,bichar_path,dim):
    char_embedding_dir = char_path
    bichar_embedding_dir = bichar_path
    char_embedding = _load_embed(char_embedding_dir,dim)
    bichar_embedding = _load_embed(bichar_embedding_dir,dim)

    embedding_matrix = _load_bichar_embedding_matrix(word_index, bichar_embedding,char_embedding,dim)

    return embedding_matrix

def get_bichar_glove(word_index,char_path,bichar_path,dim):
    char_embedding_dir = char_path
    bichar_embedding_dir = bichar_path
    char_embedding = _load_embed(char_embedding_dir,dim)
    bichar_embedding = _load_embed(bichar_embedding_dir,dim)

    embedding_matrix = _load_bichar_embedding_matrix(word_index, bichar_embedding,char_embedding,dim)

    return embedding_matrix

def get_char_elmoembedding(word_index,dim):

    char_embedding_dir = './inputs/embedding_matrix/{}dim/DaGuanElmoPretrainEmbedding_{}dim.txt'.format(dim,dim)

    char_embedding = _load_embed(char_embedding_dir,dim)

    embedding_matrix = _load_embedding_matrix(word_index,char_embedding,dim)


    return embedding_matrix


def load_embedding(char_index,bichar_index,dim,dim2 = None):

    if dim2 == None:

        word2vec_path = './inputs/embedding_matrix/{}dim/DaGuanWord2Vec_{}dim.txt'.format(dim,dim)
        glove_path = './inputs/embedding_matrix/{}dim/DaGuanFastText_{}dim.txt'.format(dim,dim)
        fasttext_path = './inputs/embedding_matrix/{}dim/DaGuanFastText_{}dim.txt'.format(dim,dim)

        bichar_word2vec_path = './inputs/embedding_matrix/{}dim/DaGuanBicharWord2Vec_{}dim.txt'.format(dim,dim)
        bichar_glove_path = './inputs/embedding_matrix/{}dim/DaGuanBicharGlove_{}dim.txt'.format(dim,dim)
        bichar_fasttext__path = './inputs/embedding_matrix/{}dim/DaGuanBicharFastText_{}dim.txt'.format(dim,dim)

        word2vec = get_word2vec(char_index,word2vec_path,dim)
        glove = get_glove(char_index,glove_path,dim)
        fasttext = get_fasttext(char_index,fasttext_path,dim)

        bichar_word2vec = get_bichar_word2vec(bichar_index,word2vec_path,bichar_word2vec_path,dim)
        bichar_glove = get_bichar_glove(bichar_index,glove_path,bichar_glove_path,dim)
        bichar_fasttext = get_bichar_fasttext(bichar_index,fasttext_path,bichar_fasttext__path,dim)

    else:

        word2vec_path = './inputs/embedding_matrix/{}dim/DaGuanWord2Vec_{}dim.txt'.format(dim, dim)
        glove_path = './inputs/embedding_matrix/{}dim/DaGuanFastText_{}dim.txt'.format(dim, dim)
        fasttext_path = './inputs/embedding_matrix/{}dim/DaGuanFastText_{}dim.txt'.format(dim, dim)


        _word2vec_path = './inputs/embedding_matrix/{}dim/DaGuanWord2Vec_{}dim.txt'.format(dim2, dim2)
        _glove_path = './inputs/embedding_matrix/{}dim/DaGuanFastText_{}dim.txt'.format(dim2, dim2)
        _fasttext_path = './inputs/embedding_matrix/{}dim/DaGuanFastText_{}dim.txt'.format(dim2, dim2)


        bichar_word2vec_path = './inputs/embedding_matrix/{}dim/DaGuanBicharWord2Vec_{}dim.txt'.format(dim2, dim2)
        bichar_glove_path = './inputs/embedding_matrix/{}dim/DaGuanBicharGlove_{}dim.txt'.format(dim2, dim2)
        bichar_fasttext__path = './inputs/embedding_matrix/{}dim/DaGuanBicharFastText_{}dim.txt'.format(dim2, dim2)


        word2vec = get_word2vec(char_index, word2vec_path, dim)
        glove = get_glove(char_index, glove_path, dim)
        fasttext = get_fasttext(char_index, fasttext_path, dim)

        bichar_word2vec = get_bichar_word2vec(bichar_index, _word2vec_path, bichar_word2vec_path, dim2)
        bichar_glove = get_bichar_glove(bichar_index, _glove_path, bichar_glove_path, dim2)
        bichar_fasttext = get_bichar_fasttext(bichar_index, _fasttext_path, bichar_fasttext__path, dim2)

    return word2vec,glove,fasttext,bichar_word2vec,bichar_glove,bichar_fasttext

