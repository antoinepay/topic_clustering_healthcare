# script used for tfidf:

from embeddings import  preprocessing as prep

#preprocess the data:
df = prep.full_preprocessing()
print(df.head())