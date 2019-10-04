from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np


# website to look @ for potential info: https://www.freecodecamp.org/news/how-to-process-textual-data-using-tf-idf-in-python-cd2bbc0a94a3/
# what does tokenizer do and do we need it?

# on processing nouns

def perform_tfidf(df,col,num_words):
    vectorizer = TfidfVectorizer()
    response = vectorizer.fit_transform((df.col))

    feature_names_en = np.array(vectorizer.get_feature_names())

    df_sklearn_total_en = pd.DataFrame(response.todense(), columns=feature_names_en)

    return df_sklearn_total_en




abstracts = pd.read_csv('data/abstracts_preproc.csv')
abstracts.columns

abstracts.nouns_lemmatized_text

vectorizer = TfidfVectorizer()
response = vectorizer.fit_transform((abstracts.nouns_lemmatized_text))

feature_names = np.array(vectorizer.get_feature_names())

df = pd.DataFrame(response.todense(), columns=feature_names_en)
df_T = df.T




def get_weight(doc_no):
   #  extract the column of the doc you want
   return (df_T.iloc[:,doc_no])


def get_embedding(words_to_embed):
   '''input: liste de mot
   output: array vecteur correspondants aux embed (n_mots x n_dimensions)
   '''
   return(output)

a = np.ones((29108,208))
a


def weighted_sum(doc_no,vectors):
   W=get_weight(doc_no)
   #vectors = get_embedding(np.array(W.index))

   for word in range(vectors.shape[0]):
       # multiply weight by embedding, same shape as embedding
       vectors[word, :] = np.array(W)[word]*vectors[word, :]


   #  now sum all the words in the document
   sum_all = (vectors.sum(axis=1))/(np.array(W).sum())
   return sum_all




#
# # dont need tokenizer parameter bc we've already tokenized
# vectorizer = TfidfVectorizer()
# response = vectorizer.fit_transform((abstracts.tokens))
#
#
# feature_names_en = np.array(vectorizer.get_feature_names())
# sorted_tfidf_index_en = response.max(0).toarray()[0].argsort()
#
#
# print('Smallest tfidf en:\n{}\n'.format(feature_names_en[sorted_tfidf_index_en[:10]]))
# print('Largest tfidf en: \n{}'.format(feature_names_en[sorted_tfidf_index_en[:-11:-1]]))
#
# df_sklearn_total_en = pd.DataFrame(response.todense(), columns = feature_names_en)
# df_sklearn_total_en.head()
#
# # Build a bar chart:
# df_sklearn_mean_en = df_sklearn_total_en.mean().sort_values(ascending=False).to_frame(name='tfidf mean')
# df_sklearn_mean_en.head()
# df_sklearn_mean_en[:20].plot.bar()

# cluster based on TFIDF

# Building a WordCloud



# 100 most important words par docs




