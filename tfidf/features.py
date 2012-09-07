import logging
import cPickle
import math
import cdec.sa
import tfidf
from gensim.matutils import sparse2full

score, train_tfidf = None, None
@cdec.sa.configure
def configure(config):
    global score, train_tfidf
    score = tfidf.scorer(config['tfidf_model'], config['tfidf_dict'])
    logging.info('Loading training corpus TF-IDF vectors')
    with open(config['train_tfidf']) as fp:
        train_tfidf = cPickle.load(fp)
    logging.info('Read vectors for %d sentences | vocabulary size = %d',
            train_tfidf.shape[1], train_tfidf.shape[0])

@cdec.sa.annotator
def tfidf_vector(sentence):
    global score
    return sparse2full(score(sentence[1:-1]), train_tfidf.shape[0])

@cdec.sa.feature
def CosineSimilarity(ctx):
    input_vector = ctx.meta['tfidf_vector']
    similarities = []
    for tup in ctx.matches:
        match_vector = train_tfidf[:,ctx.f_text.get_sentence_id(tup[0])]
        similarities.append(math.log(1e-6+match_vector.T.dot(input_vector)[0]))
    return sum(similarities)/len(similarities)
