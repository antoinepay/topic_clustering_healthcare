from sklearn.feature_extraction.text import TfidfVectorizer


# website to look @ for potential info: https://www.freecodecamp.org/news/how-to-process-textual-data-using-tf-idf-in-python-cd2bbc0a94a3/

# what does tokenizer do and do we need it?
vectorizer = TfidfVectorizer(tokenizer=word_tokenize)