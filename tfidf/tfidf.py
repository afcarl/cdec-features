from gensim.models import TfidfModel
from gensim.corpora.dictionary import Dictionary

def scorer(model, dic):
    tfidf = TfidfModel.load(model)
    dictionary = Dictionary.load(dic)
    def score(words):
        return tfidf[dictionary.doc2bow(words)]
    return score
