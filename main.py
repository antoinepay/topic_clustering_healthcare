# Libraries

import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine

from collections import Counter

import matplotlib.pyplot as plt
import itertools

from embeddings import Bert, BioWordVec, ELMo, GoogleSentence, Word2Vec
from repository.abstracts import load_data
from repository.preprocessing import launch_preprocessing

# Constants

random_state = 42

abstracts_path = 'data/CS2_Article_Clustering.xlsx'

# Core functions


def embed_abstract(abstracts, embedding_type):

    if embedding_type == "word2vec":
        vectors, output_format = Word2Vec().embed_text(abstracts.word_tokens)

    elif embedding_type == "biowordvec":
        vectors, output_format = BioWordVec().embed_text(abstracts.word_tokens)

    elif embedding_type == "google_sentence":
        vectors, output_format = GoogleSentence().embed_text(abstracts.sentence_tokens)

    elif embedding_type == "elmo":
        vectors, output_format = ELMo().embed_text(abstracts.word_tokens)

    elif embedding_type == "bert":
        vectors, output_format = Bert().embed_text(abstracts.word_tokens)

    else:
        raise Exception("Embedding type should be word2vec, biowordvec, google_sentence, elmo or bert")

    return vectors, output_format


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

    clusters["title_clean_lemmatized"] = abstracts.title_clean_lemmatized.values
    clusters["word_tokens_lemmatized"] = abstracts.word_tokens_lemmatized.values

    labelled_clusters = []

    for k in range(n_clusters):
        cluster = clusters.loc[clusters.cluster == k, :]
        words = [y for x in itertools.chain(cluster.title_clean_lemmatized) for y in x]
        most_common_words = Counter(words).most_common(5)
        print(k)
        print(most_common_words)
        most_common_words = [word[0] for word in most_common_words]
        cluster["labels"] = pd.Series([most_common_words] * len(cluster)).values
        labelled_clusters.append(cluster)

    return pd.concat(labelled_clusters, axis=0)


def evaluate_clusters(labelled_clusters):

    embedded_category = np.array(BioWordVec.embed_text(labelled_clusters.word_tokens_lemmatized)[0])
    embedded_labels = np.array(BioWordVec.embed_text(labelled_clusters.labels)[0])

    similarity_vector = []

    for i in range(len(embedded_labels)):
        similarity_vector.append(cosine(embedded_category[i], embedded_labels[i]))

    return np.sqrt(sum([a**2 for a in similarity_vector])/len(similarity_vector))


def nb_categories_in_clusters(labelled_clusters, n_clusters):
    return len(labelled_clusters.groupby(["cluster", "category"]).count())/n_clusters


# main

abstracts = load_data(abstracts_path=abstracts_path, with_preprocess=True)
vectors, output_format = embed_abstract(abstracts, "google_sentence")

# Modeling

from modeling import KMeansModel

kmeans_model = KMeansModel()

params = [{'n_clusters': i} for i in range(10, 40, 2)]

kmeans_model.plot_elbow(features=vectors, params=params)



clusters = make_kmeans(vectors, 15)
plot_kmeans(clusters)
labelled_clusters = label_clusters(clusters, 15, abstracts)
rmse = evaluate_clusters(labelled_clusters)
nb_categories_in_clusters(labelled_clusters, 15)

