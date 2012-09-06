import nltk
import re

STOP = set(nltk.corpus.stopwords.words('german'))

wRE = re.compile('^[^\W\d_]+$', re.UNICODE)

def filter(words):
    return [word for word in words
            if word not in STOP
            and wRE.match(word.decode('utf8'))]

def get_content(sentence):
    return filter(sentence.split())
