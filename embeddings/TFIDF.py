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



def weighted_sum(doc_no,vec= vectors):
   W=get_weight(doc_no)
   #vectors = get_embedding(np.array(W.index))

   #  is it shape(1) or shape(0)?
   for word in range(vec.shape[0]):
       # multiply weight by embedding, same shape as embedding
       vec[word] = np.array(W)[word]*vec[word]


   #  now sum all the words in the document
   sum_all = (vec.sum(axis=1))/(np.array(W).sum())
   return sum_all



abstracts = pd.read_csv('data/abstracts_preproc.csv')
abstracts.columns

df_tfidf = perform_tfidf(abstracts.nouns_lemmatized_text)


# for the embeddings we have: vectors and output_format
print(vectors)
print(output_format)

get_weight(0)[0]
vectors[0]

vectors