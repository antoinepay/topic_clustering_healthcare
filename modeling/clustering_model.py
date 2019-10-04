from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt


class ClusteringModel:

    def __init__(self, model_type):
        self.model_type = model_type
        self.model = BaseEstimator()

    def set_model_parameters(self, **params):
        self.model.set_params(**params)

    def perform_clustering(self, features):
        self.model.fit(features)

    def evaluate_clustering(self, labelled_clusters):
        pass

    def plot_elbow(self, features, params):

        #if not hasattr(self.model, 'inertia'):
        #    raise Exception('Missing inertia_ attribute to plot elbow graph')

        inertia = []

        for param_set in params:
            self.model.set_params(**param_set)
            self.model.fit(features)
            inertia.append(self.model.inertia_)

        plt.plot(range(len(params)), inertia)
        plt.show()

