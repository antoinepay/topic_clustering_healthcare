# Libraries

import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from collections import Counter

import matplotlib.pyplot as plt
import itertools

from embeddings import biowordvec, elmo, google_sentence, word2vec
from repository.abstracts import load_data
from repository.preprocessing import preprocessing

# Constants

random_state = 42

abstracts_path = 'data/CS2_Article_Clustering.xlsx'

# Core functions


def plot_inertia(vectors):
    """
    :param vectors: df with embedded text
    :return: plot of inertia
    """

    inertia = []

    for k in range(10, 50):
        kmeans = KMeans(n_clusters=k, random_state=random_state).fit(vectors)
        inertia.append(kmeans.inertia_)

    plt.plot(range(10, 50), inertia)
    plt.show()


def make_kmeans(vectors, n_clusters):
    """
    :param vectors: df with embedded text
    :param n_clusters: int number of clusters
    :return: df with the vectors and the associated clusters
    """

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(vectors)

    clusters = pd.concat(
        [
            vectors,
            pd.DataFrame(
                [i for i in kmeans.predict(vectors)],
                columns=('cluster',))
        ],
        axis=1
    )

    return clusters


def plot_kmeans(clusters):
    """
    :param clusters: df with embedded text and associated cluster
    :return: clusters plot on two first PCA dimensions
    """

    vectors = clusters.drop(["cluster"], axis=1)

    pca = pd.DataFrame(PCA(n_components=2, random_state=random_state).fit_transform(
        StandardScaler().fit_transform(
            vectors
        )
    ), columns=('dim_1', 'dim_2'))

    clusters = pd.concat(
        [pca, clusters], axis=1
    )

    plt.scatter(clusters.dim_1, clusters.dim_2, c=clusters.cluster, alpha=0.8)
    plt.show()


def label_clusters(clusters, n_clusters):
    clusters["Tiltle"] = abstracts.dropna(subset=["text", "Tiltle"]).Tiltle
    clusters["title_tokens"] = clusters["Tiltle"].apply(preprocessing)
    for k in range(n_clusters):
        cluster = clusters.loc[clusters.cluster == k, :]
        words = [y for x in itertools.chain(cluster.title_tokens) for y in x]
        most_common_words = Counter(words).most_common(5)
        print(k)
        print(list(most_common_words))


# main

abstracts = load_data(abstracts_path=abstracts_path, with_preprocess=True)

# word2vec

word2vec_embedding, output_format = word2vec.embed_text(abstracts.text)


# biowordvec

biowordvec_embedding, output_format = biowordvec.embed_text(abstracts.text)


# google sentence

google_sentence_embedding, output_format = google_sentence.embed_text(abstracts.text)


# elmo

elmo_embedding, output_format = elmo.embed_text(abstracts.text)


# Modeling

plot_inertia(elmo_embedding)
clusters = make_kmeans(elmo_embedding, 20)
plot_kmeans(clusters)
label_clusters(clusters, 20)


