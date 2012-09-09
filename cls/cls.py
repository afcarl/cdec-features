import logging
import numpy as np
import cdec.sa
import cPickle
import math

cls_dict, train_classes = None, None
@cdec.sa.configure
def configure(config):
  global cls_dict, train_classes
  logging.info('Loading the classes dictionary')
  with open(config['cls_dict']) as fclsd:
    cls_dict = cPickle.load(fclsd)
  logging.info('Read %d classes', len(cls_dict))

  logging.info('Loading the classes of the training corpus')
  with open(config['train_cls']) as fclsm:
    train_classes = cPickle.load(fclsm)
  logging.info('Read classes for %d sentences', len(train_classes))

@cdec.sa.annotator
def cls_classes(sentence):
    global cls_dict
    return [int(cls_dict.get(w, 0)) for w in sentence]

def similarity(x, y):
  '''cosine similarity using sets, or Ochiai coefficient -- probably inefficient'''
  sx = frozenset(x)
  sy = frozenset(y)
  num = len(sx.intersection(sy))
  den = math.sqrt(len(sx)*len(sy))
  return num/den

@cdec.sa.feature
def AvgWordClassSimilarity(ctx):
  global train_classes
  input_classes = ctx.meta['cls_classes']
  sims = []
  for tup in ctx.matches:
    match_classes = train_classes[ctx.f_text.get_sentence_id(tup[0])]
    sims.append(similarity(input_classes, match_classes))
  return sum(sims)/len(sims)

@cdec.sa.feature
def MaxWordClassSimilarity(ctx):
  # MAYBE - one could save computation if both 'max' and 'avg' are going to be computed
  global train_classes
  input_classes = ctx.meta['cls_classes']
  max_sim = 0
  for tup in ctx.matches:
    match_classes = train_classes[ctx.f_text.get_sentence_id(tup[0])]
    sim = similarity(input_classes, match_classes)
    max_sim = max((max_sim, sim))
  return max_sim

