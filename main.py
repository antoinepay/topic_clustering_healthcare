# Libraries
import pandas as pd

from embeddings import Bert, BioWordVec, ELMo, GoogleSentence, Word2Vec

from repository.preprocessing import launch_preprocessing

from modeling import KMeansModel, DBSCANModel, AffinityPropagationModel, BirchModel, OPTICSModel, ClusterLabelsCombiner

# Constants

random_state = 42

abstracts_path = 'data/CS2_Article_Clustering.xlsx'

# Core functions


def embed_abstract(abstracts, embedding_type):

    model = None

    if embedding_type == "word2vec":
        model = Word2Vec()
        vectors, output_format = model.embed_text(abstracts.nouns_lemmatized_text)

    elif embedding_type == "word2vec_tfidf":
        vectors, output_format = Word2VecTFIDF().embed_text(abstracts.nouns_lemmatized_text)

    elif embedding_type == "biowordvec":
        model = BioWordVec()
        vectors, output_format = model.embed_text(abstracts.nouns_lemmatized_text)

    elif embedding_type == "google_sentence":
        model = GoogleSentence()
        vectors, output_format = model.embed_text(abstracts.sentence_tokens)

    elif embedding_type == "elmo":
        model = ELMo()
        vectors, output_format = model.embed_text(abstracts.nouns_lemmatized_text.apply(" ".join))

    elif embedding_type == "bert":
        model = Bert()
        vectors, output_format = model.embed_text(abstracts.nouns_lemmatized_text.apply(" ".join))

    else:
        raise Exception("Embedding type should be word2vec, biowordvec, google_sentence, elmo or bert")

    return vectors, output_format, model


abstracts = pd.read_excel(abstracts_path)
abstracts = launch_preprocessing(abstracts)

vectors_biowordvec, output_format_biowordvec, model_biowordvec = embed_abstract(abstracts, "biowordvec")

# vectors_gs, output_format_gs, model_gs = embed_abstract(abstracts, "google_sentence")

# vectors_bert, output_format_bert, model_bert = embed_abstract(abstracts, "bert")


# Modeling


# KMeans

n_clusters = 100

model_kmeans = KMeansModel(n_clusters=n_clusters)

model_kmeans.plot_elbow(features=vectors_biowordvec, range=range(10, 40, 2))

model_kmeans = model_kmeans.set_model_parameters(n_clusters=n_clusters)
clusters = model_kmeans.perform_clustering(features=vectors_biowordvec)
model_kmeans.plot_from_pca(clusters=clusters)

labelled_clusters = model_kmeans.label_clusters(clusters=clusters, abstracts=abstracts)

rmse_kmeans = KMeansModel.evaluate_clusters(embedder=model_biowordvec, labelled_clusters=labelled_clusters)

model_kmeans.nb_categories_in_clusters(labelled_clusters=labelled_clusters)


# DBSCAN

eps = 0.1
min_samples = 5

model = DBSCANModel(eps=eps, min_samples=min_samples, metric="cosine")

clusters = model.perform_clustering(features=vectors_biowordvec)
model.plot_from_pca(clusters=clusters)

labelled_clusters = model.label_clusters(clusters=clusters, abstracts=abstracts)

# OPTICS
min_samples = 20

model = OPTICSModel(min_samples = min_samples,  metric="cosine")

clusters = model.perform_clustering(features=vectors_biowordvec)
model.plot_from_pca(clusters=clusters)

labelled_clusters = model.label_clusters(clusters=clusters, abstracts=abstracts)

# Affinity Propagation

model_affinity = AffinityPropagationModel()

clusters = model_affinity.perform_clustering(features=vectors_biowordvec)
model.plot_from_pca(clusters=clusters)

labelled_clusters = model.label_clusters(clusters=clusters, abstracts=abstracts)


# Birch

model = BirchModel(n_clusters=n_clusters)

clusters = model.perform_clustering(features=vectors_biowordvec)
model.plot_from_pca(clusters=clusters)

labelled_clusters = model.label_clusters(clusters=clusters, abstracts=abstracts)


# Clusters Combiner

abstracts = pd.read_csv('data/abstracts_preproc.csv',
                        converters={
                            "nouns_lemmatized_title": lambda x: x.strip("[]").replace("'", "").split(", "),
                            "nouns_lemmatized_text": lambda x: x.strip("[]").replace("'", "").split(", ")
                        })

vectors_biowordvec = pd.read_csv('data/biowordvec_embedding.csv')
vectors_bert = pd.read_csv('data/bert_embedding.csv')

clc = ClusterLabelsCombiner([
    (KMeansModel(n_clusters=100), vectors_biowordvec),
    (AffinityPropagationModel(), vectors_biowordvec),
    (BirchModel(n_clusters=100), vectors_biowordvec),
    (KMeansModel(n_clusters=150), vectors_bert),
    (BirchModel(n_clusters=150), vectors_bert)
])

labels = clc.combine(abstracts=abstracts, number_of_tags_to_keep=5)

rmse = clc.evaluate(embedder=model_biowordvec, abstracts=abstracts)

final = pd.concat([labels.labels, abstracts], axis=1)
