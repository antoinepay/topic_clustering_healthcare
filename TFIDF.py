from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np


# website to look @ for potential info: https://www.freecodecamp.org/news/how-to-process-textual-data-using-tf-idf-in-python-cd2bbc0a94a3/
# what does tokenizer do and do we need it?

def perform_tfidf(df,col,num_words):
    vectorizer = TfidfVectorizer()
    response = vectorizer.fit_transform((df.col))

    feature_names_en = np.array(vectorizer.get_feature_names())

    df_sklearn_total_en = pd.DataFrame(response.todense(), columns=feature_names_en)

    return df_sklearn_total_en




abstracts = pd.read_excel('data/abstracts_pubmed.xlsx')
abstracts.head()


# dont need tokenizer parameter bc we've already tokenized
vectorizer = TfidfVectorizer()
response = vectorizer.fit_transform((abstracts.tokens))


feature_names_en = np.array(vectorizer.get_feature_names())
sorted_tfidf_index_en = response.max(0).toarray()[0].argsort()


print('Smallest tfidf en:\n{}\n'.format(feature_names_en[sorted_tfidf_index_en[:10]]))
print('Largest tfidf en: \n{}'.format(feature_names_en[sorted_tfidf_index_en[:-11:-1]]))

df_sklearn_total_en = pd.DataFrame(response.todense(), columns = feature_names_en)
df_sklearn_total_en.head()

# Build a bar chart:
df_sklearn_mean_en = df_sklearn_total_en.mean().sort_values(ascending=False).to_frame(name='tfidf mean')
df_sklearn_mean_en.head()
df_sklearn_mean_en[:20].plot.bar()

# cluster based on TFIDF

# Building a WordCloud



# 100 most important words par docs




