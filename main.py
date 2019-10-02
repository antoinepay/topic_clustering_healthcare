# Libraries

import os
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from embeddings import biowordvec
from embeddings import google_sentence_embedding
from embeddings import preprocessing

from collections import Counter

import matplotlib.pyplot as plt


# Constants

random_state = 42

abstracts_path = 'data/CS2_Article_Clustering.xlsx'

if not os.path.exists(abstracts_path):

    from repository.abstracts import collect_data

    # here are defined categories for which we want articles
    categories = ['cancérologie', 'cardiologie', 'gastro',
                  'diabétologie', 'nutrition', 'infectiologie',
                  'gyneco-repro-urologie', 'pneumologie', 'dermatologie',
                  'industrie de santé', 'ophtalmologie']

    # call the function collect_data to get the abstracts
    collect_data(categories).to_csv(abstracts_path)

abstracts = pd.read_excel(abstracts_path)

vectors = pd.DataFrame(biowordvec.embed_text(abstracts.dropna(
    subset=["text", "Tiltle"]
).text)[0])

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
    clusters["Tilte"] = abstracts.dropna(subset=["text", "Tiltle"]).Tilte
    clusters["title_tokens"] = clusters["Tilte"].apply(preprocessing)
    for k in range(n_clusters):
        cluster = clusters.iloc[clusters.cluster == k, :]
        titles = " ".join(list(cluster.title_tokens.values))
        words = titles.split(" ")
        most_common_words = Counter(words).most_common(5)
        print(k)
        print(list(most_common_words))


plot_inertia(vectors)
clusters = make_kmeans(vectors, 20)
plot_kmeans(clusters)
label_clusters(clusters, 20)
