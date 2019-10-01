import pandas as pd
import re

abstracts = pd.read_csv('./abstracts.csv')

# abstracts.head()


re1 = r'\, attributes[^)]*\)'
re2 = r'(\[)?StringElement\('
re3 = r'\]'
re4 = r'\''
re5 = r'\, attributes[^)]*\)(,)?'

# apply function to keywords and text
# abstracts_clean = abstracts.copy()


def clean_me(x, keyword=True):
    if keyword:
        return re.compile("(%s|%s|%s|%s)" % (re1, re2, re3, re4)).sub('', x).split(',')
    else:
        return re.compile("(%s|%s|%s|%s)" % (re2, re3, re4, re5)).sub('', str(x))


abstracts['Keywords'] = abstracts['Keywords'].apply(lambda x: list(map(str.strip, clean_me(x))))

abstracts['text'] = abstracts['text'].apply(lambda x: clean_me(x, keyword=False))

