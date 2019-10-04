# Libraries

from modeling.clustering_model import ClusteringModel
from sklearn.cluster import KMeans
import numpy as np
from embeddings import BioWordVec
from scipy.spatial.distance import cosine


class KMeansModel(ClusteringModel):

    def __init__(self, n_clusters=15):
        super().__init__('kmeans')

        self.model = KMeans(n_clusters=n_clusters)

    def evaluate_clusters(labelled_clusters):
        embedded_category = np.array(BioWordVec.embed_text(labelled_clusters.word_tokens_lemmatized)[0])
        embedded_labels = np.array(BioWordVec.embed_text(labelled_clusters.labels)[0])

        similarity_vector = []

        for i in range(len(embedded_labels)):
            similarity_vector.append(cosine(embedded_category[i], embedded_labels[i]))

        return np.sqrt(sum([a ** 2 for a in similarity_vector]) / len(similarity_vector))
