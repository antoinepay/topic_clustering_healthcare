# Libraries

import pandas as pd
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from .embedder import Embedder


class BioWordVec(Embedder):

    def __init__(self):
        super().__init__('biowordvec')

        self.output_format = {
            'n_cols': 200
        }

        self.trained_model = 'bin/bio_embedding_extrinsic'

    # Core functions

    def get_vect(self, word, model):
        """
        :param word: word to be embedded
        :param model: model used to embed a word
        :return: word vector
        """
        try:
            return model.get_vector(word)
        except KeyError:
            return np.zeros((model.vector_size,))

    def sum_vectors(self, sentence, model):
        """
        :param sentence: sentence to be embedded
        :param model: model used to embed a word
        :return: vector
        """
        return sum(self.get_vect(w, model) for w in sentence)

    def word2vec_features(self, sentences, model):
        """
        :param sentences: sentences to be embedded
        :param model: model used to embed a word
        :return: numpy list of list
        """
        feats = np.vstack([self.sum_vectors(p, model) for p in sentences])
        return feats

    def embed_text(self, abstracts):
        """
        :param abstracts: pandas Series of abstracts
        :param output_format: dict specifying output format of the embedding method
        :return: embedding and associated format
        """

        word_vectors = KeyedVectors.load_word2vec_format(self.trained_model, binary=True)

        embedding = pd.DataFrame(self.word2vec_features(abstracts, word_vectors))

        return embedding, self.output_format
