# Libraries

from modelling.clutering_model import ClusteringModel
from sklearn.cluster import AffinityPropagation


class AffinityPropagationModel(ClusteringModel):

    def __init__(self):
        super().__init__('affinity')
        self.model = AffinityPropagation()

    def evaluate_clustering(self, clusters):
        pass
