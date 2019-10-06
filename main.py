# Libraries

import pandas as pd

from embeddings import Bert, BioWordVec, ELMo, GoogleSentence, Word2Vec
from repository.preprocessing import launch_preprocessing
from modeling import KMeansModel, DBSCANModel, AffinityPropagationModel, MeanShiftModel, ClusterLabelsCombiner


# Constants

random_state = 42

abstracts_path = 'data/CS2_Article_Clustering.xlsx'

# Core functions


def embed_abstract(abstracts, embedding_type):
    if embedding_type == "word2vec":
        vectors, output_format = Word2Vec().embed_text(abstracts.nouns_lemmatized_text)

    elif embedding_type == "biowordvec":
        vectors, output_format = BioWordVec().embed_text(abstracts.nouns_lemmatized_text)

    elif embedding_type == "google_sentence":
        vectors, output_format = GoogleSentence().embed_text(abstracts.sentence_tokens)

    elif embedding_type == "elmo":
        vectors, output_format = ELMo().embed_text(abstracts.nouns_lemmatized_text)

    elif embedding_type == "bert":
        vectors, output_format = Bert().embed_text(abstracts.nouns_lemmatized_text)

    else:
        raise Exception("Embedding type should be word2vec, biowordvec, google_sentence, elmo or bert")

    return vectors, output_format


abstracts = pd.read_excel(abstracts_path)
abstracts = launch_preprocessing(abstracts)

import pandas as pd
abstracts = pd.read_csv('data/abstracts_preproc.csv',
                        converters={
                            "nouns_lemmatized_title": lambda x: x.strip("[]").replace("'", "").split(", "),
                            "nouns_lemmatized_text": lambda x: x.strip("[]").replace("'", "").split(", ")
                        })

vectors, output_format = embed_abstract(abstracts, "biowordvec")

# Modeling


# KMeans

model = KMeansModel()

params = [{'n_clusters': i} for i in range(10, 40, 2)]
model.plot_elbow(features=vectors, params=params)

n_clusters = 10

model = model.set_model_parameters(n_clusters=n_clusters)
clusters = model.perform_clustering(features=vectors)
model.plot_from_pca(clusters=clusters)

labelled_clusters = model.label_clusters(clusters=clusters, abstracts=abstracts, n_clusters=n_clusters)

rmse_kmeans = model.evaluate_clusters(embedder=BioWordVec(), labelled_clusters=labelled_clusters)

model.nb_categories_in_clusters(labelled_clusters=labelled_clusters, n_clusters=n_clusters)


# DBSCAN

eps = 0.1
min_samples = 5

model = DBSCANModel(eps=eps, min_samples=min_samples, metric="cosine")

clusters = model.perform_clustering(features=vectors)
model.plot_from_pca(clusters=clusters)

labelled_clusters = model.label_clusters(clusters=clusters, abstracts=abstracts, n_clusters=n_clusters)


# Affinity Propagation

model = AffinityPropagationModel()

clusters = model.perform_clustering(features=vectors)
model.plot_from_pca(clusters=clusters)

labelled_clusters = model.label_clusters(clusters=clusters, abstracts=abstracts, n_clusters=n_clusters)


# MeanShift

model = MeanShiftModel()

clusters = model.perform_clustering(features=vectors)
model.plot_from_pca(clusters=clusters)

labelled_clusters = model.label_clusters(clusters=clusters, abstracts=abstracts, n_clusters=n_clusters)


# Clusters Combiner

clc = ClusterLabelsCombiner([
    (model, vectors),
    (model, vectors)
])

clc.combine(abstracts=abstracts, n_clusters=10, number_of_tags_to_keep=5)