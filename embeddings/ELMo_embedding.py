### not formatted yet


#Tutorial:
# https://medium.com/saarthi-ai/elmo-for-contextual-word-embedding-for-text-classification-24c9693b0045


#import packages
import pandas as pd
import tensorflow_hub as hub
import tensorflow as tf


def elmo_sentence_embedding(series_abtracts):
    """
    :param series_abstracts: pandas Series of abstracts
    """

    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

    list_abstracts = list(series_abstracts.values)

    embeddings = elmo(list_abstracts, signature="default", as_dict=True)["elmo"]

    return embeddings



### test on series of abstracts:
abstracts= ["Abstract1: Hello my name is Rick", "Abstract2: your name is Tom","Abstract3: I want to eat an apple"]
df = pd.DataFrame()
df["abstracts"] = abstracts
series_abstracts = pd.Series(df["abstracts"])

elmo_sentence_embedding(series_abstracts)


