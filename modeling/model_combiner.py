# Libraries
import pandas as pd
import numpy as np
from collections import Counter


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

        self.clusters = self.fit_models()

    def fit_models(self):
        """
        :return: list with fitted models associated with the used embedding
        """
        return [model.perform_clustering(vectors) for model, vectors in self.models_vectors_mapping]

    def get_clusters_from_models(self, abstracts):
        """
        :return: list of pandas Series with predicted clusters for each embedding
        """
        return [pd.Series(model.label_clusters(
            clusters=self.clusters[i],
            abstracts=abstracts
        ).labels) for i, (model, vectors) in enumerate(self.models_vectors_mapping)]

    def concat_labels(self, labels):
        """
        Concat a list of multiple labels columns to only one by adding labels in a single group
        :param labels: list of labels columns
        :return: Series, list of a labels column
        """
        result = labels[0]

        for l in range(1, len(labels)):
            result += labels[l]

        return result

    def tfidf(self, labels, number_of_tags_to_keep=5):
        """
        :param labels: Series of labels
        :param number_of_tags_to_keep: Number of tags to keep after the tfidf process
        :return: List of the most relevant labels
        """

        counters = []

        # Term Frequency

        for words in labels.tolist():
            c = Counter(words)
            for k in c.keys():
                c[k] = c[k] / len(words)
            counters.append(c)

        # Inverse Document Frequency

        docs_with_word = Counter()

        for i, l in enumerate(labels.tolist()):
            for word in counters[i].keys():
                docs_with_word[word] += 1

        for k in docs_with_word.keys():
            docs_with_word[k] = np.log(labels.shape[0] / docs_with_word[k])

        # TF-IDF

        for counter in counters:
            for k in counter.keys():
                counter[k] = counter[k] * docs_with_word[k]

        most_relevant_words = []

        for counter in counters:
            most_relevant_words.append([word[0] for word in counter.most_common(number_of_tags_to_keep)])

        return most_relevant_words

    def combine(self, abstracts, number_of_tags_to_keep=5):
        """
        Final function to call
        :return:
        """
        clusters = self.get_clusters_from_models(abstracts=abstracts)

        labels = self.concat_labels(clusters)

        return self.tfidf(labels=labels, number_of_tags_to_keep=number_of_tags_to_keep)