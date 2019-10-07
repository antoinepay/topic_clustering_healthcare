
import pandas as pd
from sklearn.cluster import OPTICS


from modeling.clustering_model import ClusteringModel


class OPTICSModel(ClusteringModel):

    def __init__(self, **params):
        super().__init__('optics')

        self.model = OPTICS(**params)

