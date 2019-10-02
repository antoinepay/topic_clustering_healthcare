# Libraries
import os
import pandas as pd
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

# Constants

OUTPUT_FORMAT = {
    'n_cols': 200
}

TRAINED_MODEL = 'bin/bio_embedding_extrinsic'


# Core functions

def get_vect(word, model):
    """
    :param word: word to be embedded
    :param model: model used to embed a word
    :return: word vector
    """
    try:
        return model.get_vector(word)
    except KeyError:
        return np.zeros((model.vector_size,))


def sum_vectors(sentence, model):
    """
    :param sentence: sentence to be embedded
    :param model: model used to embed a word
    :return: vector
    """
    return sum(get_vect(w, model) for w in sentence)


def word2vec_features(sentences, model):
    """
    :param sentences: sentences to be embedded
    :param model: model used to embed a word
    :return: numpy list of list
    """
    feats = np.vstack([sum_vectors(p, model) for p in sentences])
    return feats


def embed_text(abstracts, output_format=None):
    """
    :param abstracts: pandas Series of abstracts
    :param output_format: dict specifying output format of the embedding method
    :return: embedding and associated format
    """

    if output_format is None:
        output_format = OUTPUT_FORMAT

    word_vectors = KeyedVectors.load_word2vec_format(TRAINED_MODEL, binary=True)

    embedding = pd.DataFrame(word2vec_features(abstracts, word_vectors))

    return embedding, output_format
