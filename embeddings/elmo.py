# Libraries

import pandas as pd
import tensorflow_hub as hub
import tensorflow as tf


# Tutorial:

# https://medium.com/saarthi-ai/elmo-for-contextual-word-embedding-for-text-classification-24c9693b0045


OUTPUT_FORMAT = {
    'n_cols': 1024
}


def embed_text(abstracts, output_format=None):
    """
    :param abstracts: pandas Series of abstracts
    :param output_format: dict specifying output format of the embedding method
    :return: embedding and associated format
    """

    if output_format is None:
        output_format = OUTPUT_FORMAT

    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

    embeddings = elmo(abstracts.tolist(), signature="default", as_dict=True)["elmo"]

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.tables_initializer())
        # return average of ELMo features
        return sess.run(tf.reduce_mean(embeddings, 1)), output_format
