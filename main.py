# Libraries

import os
import pandas as pd

from sklearn.cluster import KMeans

from embeddings import biowordvec
from embeddings import google_sentence_embedding


# Constants

random_state = 42

abstracts_path = 'data/abstracts.csv'

if not os.path.exists(abstracts_path):

    from repository.abstracts import collect_data

    # here are defined categories for which we want articles
    categories = ['cancérologie', 'cardiologie', 'gastro',
                  'diabétologie', 'nutrition', 'infectiologie',
                  'gyneco-repro-urologie', 'pneumologie', 'dermatologie',
                  'industrie de santé', 'ophtalmologie']

    # call the function collect_data to get the abstracts
    collect_data(categories).to_csv(abstracts_path)

abstracts = pd.read_csv(abstracts_path)


# Core functions

def make_kmeans(embedding_type, n_clusters):
    """
    :param embedding_type: str either 'biowordvec' or 'guse'
    :param n_clusters: int number of clusters
    :return:
    """

    if embedding_type == "biowordvec":
        vectors, format = biowordvec.embed_text(abstracts.dropna().text)
    elif embedding_type == "guse":
        vectors, format = google_sentence_embedding.embed_text(abstracts.dropna().text)
    else:
        raise Exception("embedding_type should be either biowordvec or guse")

    vectors = pd.DataFrame(vectors.apply(pd.Series))

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(vectors)

    clusters = pd.concat(
        [
            vectors,
            pd.DataFrame(
                [str(i) for i in kmeans.predict(vectors)],
                columns=('cluster',))
        ],
        axis=1
    )

    return clusters

