from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np


# website to look @ for potential info: https://www.freecodecamp.org/news/how-to-process-textual-data-using-tf-idf-in-python-cd2bbc0a94a3/
# what does tokenizer do and do we need it?

# on processing nouns

def perform_tfidf(df_col):
    vectorizer = TfidfVectorizer()
    response = vectorizer.fit_transform((df_col))

    feature_names_en = np.array(vectorizer.get_feature_names())

    df_tfidf = pd.DataFrame(response.todense(), columns=feature_names_en)

    return df_tfidf


def get_weight(doc_no):
   #  extract the column of the doc you want
   return (df_tfidf.T.iloc[:,doc_no])


def get_embedding(words_to_embed):
   '''input: liste de mot
   output: array vecteur correspondants aux embed (n_mots x n_dimensions)
   '''
   return(output)



def weighted_sum(doc_no,vectors):
   W=get_weight(doc_no)
   #vectors = get_embedding(np.array(W.index))

   for word in range(vectors.shape[0]):
       # multiply weight by embedding, same shape as embedding
       vectors[word] = np.array(W)[word]*vectors[word]


   #  now sum all the words in the document
   sum_all = (vectors.sum(axis=1))/(np.array(W).sum())
   return sum_all



abstracts = pd.read_csv('data/abstracts_preproc.csv')
abstracts.columns

df_tfidf = perform_tfidf(abstracts.nouns_lemmatized_text)


# for the embeddings we have: vectors and output_format
print(vectors)
print(output_format)