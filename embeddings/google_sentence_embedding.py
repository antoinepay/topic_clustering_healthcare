# Libraries
import pandas as pd
import numpy as np

import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Constants

OUTPUT_FORMAT = {
    'n_cols': 512
}

MODULE = "https://tfhub.dev/google/universal-sentence-encoder-large/3"

# Core functions


def embed_useT(module):
    with tf.Graph().as_default():
        sentences = tf.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sentences)
        session = tf.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})


def embed_text(abstracts, output_format=None):
    """
    :param abstracts: pandas Series of abstracts
    :param output_format: dict specifying output format of the embedding method
    :return: embedding and associated format
    """

    if output_format is None:
        output_format = OUTPUT_FORMAT

    sentences = list(abstracts)

    embed_fn = embed_useT(MODULE)

    embedding = embed_fn(sentences)

    return embedding[0], output_format

