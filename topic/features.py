import math
from itertools import izip
import cdec.sa
from lda import LDA

lda = None
@cdec.sa.configure
def configure(config):
    global lda
    lda = LDA('model.txt', 'vocab.txt')
    lda.load()

@cdec.sa.annotator
def lda_topics(sentence):
    return lda.topic_vector(sentence[1:-1])

def divergence(x, y):
    return sum(px*math.log(px/py) for px, py in izip(x, y))

@cdec.sa.feature
def lda_similarity(ctx):
    input_topics = ctx.meta['lda_topics']
    divs = []
    for tup in ctx.matches:
        match = ctx.f_text.get_sentence(ctx.f_text.get_sentence_id(tup[0]))[:-1]
        match_topics = lda.topic_vector(match)
        divs.append(divergence(input_topics, match_topics))
    return sum(divs)/len(divs)
