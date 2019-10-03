#import packages
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.stem.wordnet import WordNetLemmatizer


#mandatory downloads
def nltk_package_downloads():
    nltk.download('punkt')
    nltk.download('stopwords')


def full_preprocessing():
    """
    :return: a preproceessed data frame
    """
    df = pd.read_excel("../data/CS2_Article_Clustering.xlsx")
    df = df.rename(columns = {"Tiltle": "title"})
    df.dropna(subset=['text', 'title'],inplace=True)

    df["tokens"] = df["text"].apply(preprocessing)
    df["title_clean"] = df["title"].apply(preprocessing)

    final = detokenize(df,"tokens")

    return final

def preprocessing(column):
    """
    :param text_column:
    :return: a pre-processed column
    """

    # Tokenization
    tokens = word_tokenize(str(column))

    # Deleting words with  only one caracter
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
                          "6", "7", "8", "9", "'", '.', '‘', ';']
    transformation_sc_dict = {initial: " " for initial in special_characters}
    tokens = [token.translate(str.maketrans(transformation_sc_dict)) for token in tokens]
    return tokens


def detokenize(df_tokens,name_token_column):
    """
    :param df_tokens: dataframe with a tokenized column that will be detokenized
    :param name_token_column: string with the name of the column holding the tokenized data
    :return: the same dataframe with an additional detokenized column called "detokenized"
    the detokenization turns [early, phase, trials, institut, oncology] into early phase therapeutic trials oncology
    """
    df_tokens["detokenized"] = df_tokens[name_token_column].apply(TreebankWordDetokenizer().detokenize)
    return df_tokens


#testing_main (needs to be called in the actual main):
#nltk_package_downloads()
#df = read_data_df("data/CS2_Article_Clustering.xlsx")
#df = delete_empty_rows(df)
#df["tokens"] = df["text"].apply(preprocessing)

#df = full_preprocessing()
#print(df.head())
# df.to_excel("../data/abstracts_pubmed.xlsx",index=False)
