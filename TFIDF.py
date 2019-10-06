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

    return df_tfidf.T


def get_weight(doc_no):
   #  extract the column of the doc you want
   return (df_tfidf.iloc[:,doc_no])


def get_embedding(words_to_embed):
   '''input: liste de mot
   output: array vecteur correspondants aux embed (n_mots x n_dimensions)
   '''
   return(output)



def weighted_sum(doc_no, vec=vectors):
   W=get_weight(doc_no)
   print(np.array(W).shape)
   #vectors = get_embedding(np.array(W.index))


   for word in range(vec.shape[0]):
       # multiply weight by embedding, same shape as embedding
       vec[word] = np.array(W)[word]*vec[word]

   #  now sum all the words in the document
   sum_all = (vec.sum(axis=1))/(np.array(W).sum())
   return sum_all


abstracts = pd.read_csv('data/abstracts_preproc.csv')
abstracts.columns

df_tfidf = perform_tfidf(abstracts.nouns_lemmatized_text)
df_tfidf.head()

abstracts.head

# construct a dataframe of summed vectors for each document. Document in column, vect param in rows.
# get the article_ID
# populate dataframe, add row each iter, new doc


# for the embeddings we have: vectors and output_format
print(vectors)
print(output_format)

d = []
for index, row in abstracts.head().iterrows():
    d.append({'article_ID': row.article_ID, 'vec_tfidf': weighted_sum(index,vectors)})
pd.DataFrame(d)

weighted_sum(0)
get_weight(0)
vectors

# diff bw weight and vectors?
# get_weight(0) : all the weights in doc 0 of ALL the words   len: 29108
# vectors: embedding, 6909 articles, 299 columns









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




