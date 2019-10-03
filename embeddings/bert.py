# Libraries

import pandas as pd
from bert_embedding import BertEmbedding

# Constants

OUTPUT_FORMAT = {
    'n_cols': 200
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

    bert = BertEmbedding(model='bert_12_768_12', dataset_name='book_corpus_wiki_en_uncased')
    bert_embedding = bert(abstracts.tolist(), oov_way='sum')

    embedding = []

    for _, vectors in bert_embedding:
        embedding.append(sum(vectors))

    embedding = pd.DataFrame(embedding)

    return embedding, output_format
