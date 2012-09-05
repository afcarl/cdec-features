import nltk
import re

STOP = set(nltk.corpus.stopwords.words('german'))

wRE = re.compile('^[^\W\d_]+$', re.UNICODE)

def get_content(sentence):
    return [word for word in sentence.split()
            if word not in STOP
            and wRE.match(word.decode('utf8'))]
