# Libraries
import pandas as pd

from embeddings import Bert, BioWordVec, ELMo, GoogleSentence, Word2Vec
from embeddings import Word2VecTFIDF

from repository.preprocessing import launch_preprocessing

from modeling import KMeansModel, DBSCANModel, AffinityPropagationModel, MeanShiftModel, OPTICSModel, ClusterLabelsCombiner


# Constants

random_state = 42

abstracts_path = 'data/CS2_Article_Clustering.xlsx'

# Core functions


def embed_abstract(abstracts, embedding_type):
    if embedding_type == "word2vec":
        vectors, output_format = Word2Vec().embed_text(abstracts.nouns_lemmatized_text)

    elif embedding_type == "word2vec_tfidf":
        vectors, output_format = Word2VecTFIDF().embed_text(abstracts.nouns_lemmatized_text)

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

abstracts = pd.read_csv('data/abstracts_preproc.csv',
                        converters={
                            "nouns_lemmatized_title": lambda x: x.strip("[]").replace("'", "").split(", "),
                            "nouns_lemmatized_text": lambda x: x.strip("[]").replace("'", "").split(", ")
                        })

vectors, output_format = embed_abstract(abstracts, "biowordvec")


# give word
# get df_tfidf from TFIDF.py
veco = df_tfidf.copy()
for col in veco.columns:
    veco[col].values[:] = 0
# veco.to_numpy().sum()
# check that it equals 0

vectors, output_format = embed_abstract(abstracts, "word2vec_tfidf")


# Modeling


# KMeans

model_kmeans = KMeansModel()

params = [{'n_clusters': i} for i in range(10, 40, 2)]
model_kmeans.plot_elbow(features=vectors, params=params)

n_clusters = 100

model_kmeans = model_kmeans.set_model_parameters(n_clusters=n_clusters)
clusters = model_kmeans.perform_clustering(features=vectors)
model_kmeans.plot_from_pca(clusters=clusters)

labelled_clusters = model_kmeans.label_clusters(clusters=clusters, abstracts=abstracts, n_clusters=n_clusters)

rmse_kmeans = model_kmeans.evaluate_clusters(embedder=BioWordVec(), labelled_clusters=labelled_clusters)

model_kmeans.nb_categories_in_clusters(labelled_clusters=labelled_clusters, n_clusters=n_clusters)


# DBSCAN

eps = 0.1
min_samples = 5

model = DBSCANModel(eps=eps, min_samples=min_samples, metric="cosine")

clusters = model.perform_clustering(features=vectors)
model.plot_from_pca(clusters=clusters)

labelled_clusters = model.label_clusters(clusters=clusters, abstracts=abstracts, n_clusters=n_clusters)

#OPTICS
min_samples = 20

model = OPTICSModel(min_samples = min_samples,  metric="cosine")

clusters = model.perform_clustering(features=vectors)
model.plot_from_pca(clusters=clusters)

labelled_clusters = model.label_clusters(clusters=clusters, abstracts=abstracts, n_clusters=n_clusters)

# Affinity Propagation

model_affinity = AffinityPropagationModel()

clusters = model_affinity.perform_clustering(features=vectors)
model.plot_from_pca(clusters=clusters)

labelled_clusters = model.label_clusters(clusters=clusters, abstracts=abstracts, n_clusters=n_clusters)


# MeanShift

model = MeanShiftModel()

clusters = model.perform_clustering(features=vectors)
model.plot_from_pca(clusters=clusters)

labelled_clusters = model.label_clusters(clusters=clusters, abstracts=abstracts, n_clusters=n_clusters)


# Clusters Combiner

clc = ClusterLabelsCombiner([
    (model_kmeans, vectors),
    (model_affinity, vectors)
])

clc.combine(abstracts=abstracts, n_clusters=10, number_of_tags_to_keep=5)
