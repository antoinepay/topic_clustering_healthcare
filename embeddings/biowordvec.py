# Libraries


# Constants

OUTPUT_FORMAT = {
    'n_cols': 512
}


# Core functions

def embed_text(data, output_format=None, columns=None):

    if output_format is None:
        output_format = OUTPUT_FORMAT

    features = data.copy()

    if columns is not None:
        features = features[[columns]]

    embedding = ...

    """
    Embedding method
    """

    return embedding, output_format
