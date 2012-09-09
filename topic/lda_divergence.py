import logging
import numpy as np
import cPickle
import math
import cdec.sa
from lda import LDA

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

def sym_divergence(x, y):
    m = (x+y)/2
    return (divergence(x, m) + divergence(y, m))/2

def divergence(x, y):
    return math.log((x * np.log(x/y)).sum())

@cdec.sa.feature
def LDADivergence(ctx):
    input_topics = ctx.meta['lda_topics']
    divs = []
    for tup in ctx.matches:
        match_topics = train_topics[ctx.f_text.get_sentence_id(tup[0])]
        try:
            divs.append(sym_divergence(input_topics, match_topics))
        except:
            pass
    if len(divs):
        return sum(divs)/len(divs)
    else:
        return 0
