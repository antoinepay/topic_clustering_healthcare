# Libraries

from modeling.clutering_model import ClusteringModel
from sklearn.cluster import KMeans


class KMeansModel(ClusteringModel):

    def __init__(self, n_clusters=15):
        super().__init__('kmeans')

        self.model = KMeans(n_clusters=n_clusters)
