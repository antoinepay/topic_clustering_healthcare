import pandas as pd
import xlrd
import numpy as np
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



abstracts = pd.read_excel('data/CS2_Article_Clustering.xlsx')

# all columns lower case:
abstracts1 = abstracts.copy()

abstracts1.columns

abstracts['text'] = abstracts['text'].str.lower()
abstracts['Tiltle'] = abstracts['Tiltle'].str.lower()
abstracts['Keywords'] = abstracts['Keywords'].str.lower()


text = abstracts['text'][0]
wordcloud = WordCloud().generate(text)


plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()




%time
1+1