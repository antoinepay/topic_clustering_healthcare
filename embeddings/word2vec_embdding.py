
#import packages

import gensim
import  numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors


OUTPUT_FORMAT = {
    'n_cols': 300
}


def get_vector_representation(word, output_format=None):

    if output_format is None:
        output_format = OUTPUT_FORMAT

    path_to_google_vectors = ""
    w2v = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    try:
        word_array = w2v[word].reshape(1,-1)
        return word_array
    except :
        # if word not in google word2vec vocabulary, return vector with low norm
        return np.zeros((1,300))





