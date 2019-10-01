# Libraries

import os
import pandas as pd

from embeddings import biowordvec

abstracts_path = 'data/abstracts.csv'

if not os.path.exists(abstracts_path):

    from repository.abstracts import collect_data

    # here are defined categories for which we want articles
    categories = ['cancérologie', 'cardiologie', 'gastro',
                  'diabétologie', 'nutrition', 'infectiologie',
                  'gyneco-repro-urologie', 'pneumologie', 'dermatologie',
                  'industrie de santé', 'ophtalmologie']

    # call the function collect_data to get the abstracts
    collect_data(categories).to_csv(abstracts_path)

abstracts = pd.read_csv(abstracts_path)

vectors, format = biowordvec.embed_text(abstracts.dropna().text)

