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
    df = read_data_df("../data/CS2_Article_Clustering.xlsx")
    df = delete_empty_rows(df)
    df["tokens"] = df["text"].apply(preprocessing)
    final = detokenize(df,"tokens")

    return final


def read_data_df(path):
    """
    :param path: path to the data
    :return: dataframe with the file
    """
    df = pd.read_excel(path)
    return df

def delete_empty_rows(df):
    """
    :param df: the entire dataframe
    :return: a dataframe with deleted rows (which were empty)
    """
    ind = df["text"].isna()
    df = df[-ind]
    return df

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
    stopW = stopwords.words('english')
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
    :return: the same dataframe with an additional detkoenized column called "detokenized"
    """
    df_tokens["detokenized"] = df_tokens[name_token_column].apply(TreebankWordDetokenizer().detokenize)
    return df_tokens



#df = full_preprocessing()
#print(df.head())
