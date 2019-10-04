# Libraries

import pandas as pd


class ClusterLabelsCombiner:

    def __init__(self, models_vectors_mapping):
        """
        :param models_vectors_mapping: list, association of ClusteringModel with a specific embedding

        Example: ClusterLabelsCombiner(
            [
                (KMeansModel(n_clusters=15), biowordvec_embedding),
                (DBSCANModel(eps=0.1), bert_embedding)
            ]
        )

        """
        self.models_vectors_mapping = models_vectors_mapping

        self.fit_models()

    def fit_models(self):
        """
        :return: list with fitted models associated with the used embedding
        """
        return [model.perform_clustering(vectors) for model, vectors in self.models_vectors_mapping if not hasattr(model, 'predict')]

    def get_clusters_from_models(self, abstracts, n_clusters):
        """
        :return: list of pandas Series with predicted clusters for each embedding
        """
        return [pd.Series(model.label_clusters(
            features=vectors,
            abstracts=abstracts,
            n_clusters=n_clusters
        ).labels) for model, vectors in self.models_vectors_mapping]


