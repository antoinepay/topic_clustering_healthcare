# Libraries

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from collections import Counter

import matplotlib.pyplot as plt
import itertools

from embeddings import biowordvec, elmo, google_sentence, word2vec, bert
from repository.abstracts import load_data
from repository import preprocessing

# Constants

random_state = 42

abstracts_path = 'data/CS2_Article_Clustering.xlsx'

# Core functions


def embed_abstract(abstracts, embedding_type):

    if embedding_type == "word2vec":
        vectors, output_format = word2vec.embed_text(abstracts.word_tokens)

    elif embedding_type == "biowordvec":
        vectors, output_format = biowordvec.embed_text(abstracts.word_tokens)

    elif embedding_type == "google_sentence":
        vectors, output_format = google_sentence.embed_text(abstracts.sentence_tokens)

    elif embedding_type == "elmo":
        vectors, output_format = elmo.embed_text(abstracts.word_tokens)

    elif embedding_type == "bert":
        vectors, output_format = bert.embed_text(abstracts.word_tokens)

    else:
        raise Exception("Embedding type should be word2vec, biowordvec, google_sentence, elmo or bert")

    return vectors, output_format


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


def concat_clusters_with_abstracts_information(clusters, abstracts, columns):
    return pd.concat([clusters, abstracts[columns]], axis=1)


def label_clusters(clusters, n_clusters, abstracts):
    """
    :param clusters: df with clusters
    :param n_clusters: int nb clusters
    :param abstracts: df with all the information
    :return labelled clusters
    """

    clusters["title_clean"] = abstracts.title_clean
    clusters["category"] = abstracts.category

    for i in clusters.index:
        if type(clusters.loc[i, "title_clean"]) is float:
            clusters.loc[i, "title_clean"] = ["nan"]

    labelled_clusters = []

    for k in range(n_clusters):
        cluster = clusters.loc[clusters.cluster == k, :]
        words = [y for x in itertools.chain(cluster.title_clean) for y in x]
        most_common_words = Counter(words).most_common(5)
        print(k)
        print(most_common_words)
        most_common_words = [word[0] for word in most_common_words]
        cluster["labels"] = [most_common_words] * len(cluster)
        labelled_clusters.append(cluster)

    return pd.concat(labelled_clusters, axis=0)


def evaluate_clusters(labelled_clusters):

    for i in labelled_clusters.index:
        if type(clusters.loc[i, "category"]) is float:
            clusters.loc[i, "category"] = ["nan"]

    embedded_category = biowordvec.embed_text(labelled_clusters.category)
    embedded_labels = biowordvec.embed_text(labelled_clusters.labels)

    return np.sqrt(mean_squared_error(embedded_category, embedded_labels))


def nb_categories_in_clusters(labelled_clusters, n_clusters):
    return len(labelled_clusters.groupby(["cluster", "category"]).count())/n_clusters


# main

abstracts = load_data(abstracts_path=abstracts_path, with_preprocess=True)
vectors = embed_abstract(abstracts, "biowordvec")[0]

# Modeling

plot_inertia(vectors)
clusters = make_kmeans(vectors, 15)
plot_kmeans(clusters)
labelled_clusters = label_clusters(clusters, 15, abstracts)
evaluate_clusters(labelled_clusters)
nb_categories_in_clusters(labelled_clusters, 15)

