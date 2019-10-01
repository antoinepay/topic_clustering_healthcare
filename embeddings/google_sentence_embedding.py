# Libraries
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import re

# Constants

OUTPUT_FORMAT = {
    'n_cols': 512
}

# Core functions


def embed_text(abstracts, output_format=None):
    """
    :param abstracts: pandas Series of abstracts
    :param output_format: dict specifying output format of the embedding method
    :return: embedding and associated format
    """

    if output_format is None:
        output_format = OUTPUT_FORMAT

    # Import the Universal Sentence Encoder's TF Hub module
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
    embed = hub.Module(module_url)

    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.ERROR)

    embedding = []

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        for sentence in abstracts.values:
            sentence = sentence.replace("!", ".").replace("?", ".").replace("...", ".").split(".")
            try:
                vector = session.run(embed(sentence))
                embedding = embedding.append(vector[0])
            except:
                embedding = embedding.append(np.zeros([output_format['n_cols']]))

    return embedding, output_format

