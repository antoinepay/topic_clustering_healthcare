import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import time


# DISCLAIMER
# DO NOT RUN unless you want to make yourself a coffee: 
# In my case:
# text takes 8mins
# title takes 14mins

# mandatory downloads
def nltk_package_downloads():
    nltk.download('punkt')
    nltk.download('stopwords')


def preprocessing(column):
    """
    :param text_column:
    :return: a pre-processed column
    """

    # Tokenization
    tokens = word_tokenize(str(column))

    # Deleting words with  only one character
    tokens = [token for token in tokens if len(token) > 2]

    # stopwords + lowercase
    normal_stopwords = stopwords.words('english')
    #     import more extensive stopwords + convert to list the first column
    comprehensive_stopwords = \
    pd.read_csv('https://raw.githubusercontent.com/Alir3z4/stop-words/master/english.txt', header=None)[0].tolist()
    #     potential further stopwords (will have to test that)
    special_medical = ["complex", "patients", "treatment", "months",
                       "rate", "prevalence", "case", "early", "management",
                       "reported", "information", "baseline", "study", "questionnaire", "results", "month", "months",
                       "years", "year"]

    stopW = normal_stopwords + comprehensive_stopwords + special_medical

    tokens = [token.lower() for token in tokens if token.lower() not in stopW]

    # Deleting specific characters
    special_characters = ["@", "/", "#", ".", ",", "!", "?", "(", ")",
                          "-", "_", "’", "'", "\"", ":", "=", "+", "&",
                          "`", "*", "0", "1", "2", "3", "4", "5",
                          "6", "7", "8", "9", "'", '.', '‘', ';', '=', "<", ">", "±", "%"]

    transformation_sc_dict = {initial: " " for initial in special_characters}
    tokens = [token.translate(str.maketrans(transformation_sc_dict)) for token in tokens]

    return tokens


# testing_main (needs to be called in the actual main):
# nltk_package_downloads()
df = pd.read_excel("../data/CS2_Article_Clustering.xlsx")
df = df.rename(columns={"Tiltle": "title"})
df.dropna(subset=['text', 'title'], inplace=True)
# start_time = time.time()
df["text_clean"] = df["text"].apply(preprocessing)
# print("--- %s seconds ---" % (time.time() - start_time))
print("now doing title_clean")
df["title_clean"] = df["title"].apply(preprocessing)
# print("--- %s seconds ---" % (time.time() - start_time))


# writing to excel in the data folder:
df.to_excel("../data/abstracts_pubmed.xlsx")