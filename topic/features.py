import math
import logging
from itertools import izip
import cdec.sa
from lda import LDA
import cPickle

lda, train_topics = None, None
@cdec.sa.configure
def configure(config):
    global lda, train_topics
    lda = LDA(config['lda_model'], config['lda_dict'])
    lda.load()
    logging.info('Loading training corpus topics')
    with open(config['train_topics']) as fp:
        train_topics = cPickle.load(fp)
    logging.info('Read topics for %d sentences', len(train_topics))

@cdec.sa.annotator
def lda_topics(sentence):
    global lda
    return lda.topic_vector(sentence[1:-1])

def divergence(x, y):
    return sum(px*math.log(px/py) for px, py in izip(x, y))

@cdec.sa.feature
def lda_similarity(ctx):
    input_topics = ctx.meta['lda_topics']
    divs = []
    for tup in ctx.matches:
        match_topics = train_topics[ctx.f_text.get_sentence_id(tup[0])]
        divs.append(divergence(input_topics, match_topics))
    return sum(divs)/len(divs)
