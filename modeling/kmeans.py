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

