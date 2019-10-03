from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


# website to look @ for potential info: https://www.freecodecamp.org/news/how-to-process-textual-data-using-tf-idf-in-python-cd2bbc0a94a3/

# what does tokenizer do and do we need it?


abstracts = pd.read_excel('data/abstracts_pubmed.xlsx')
abstracts.head()
abstracts.columns
abstracts.tokens[5]



# dont need tokenizer parameter bc we've already tokenized
vectorizer = TfidfVectorizer()
response = vectorizer.fit_transform((abstracts.tokens))


feature_names_en = np.array(vectorizer.get_feature_names())
sorted_tfidf_index_en = response.max(0).toarray()[0].argsort()

print('Smallest tfidf en:\n{}\n'.format(feature_names_en[sorted_tfidf_index_en[:10]]))

df_sklearn_total_en = pd.DataFrame(response.todense(), columns = feature_names_en)
df_sklearn_total_en.head()







# have to get rid of empty tokens
# have to get rid of p
# have to get rid of n
'    '
'   '