import numpy as np
import cPickle
import sys
from collections import defaultdict

def rewrite(txt_corpus, pickle_corpus, cls_dict):
  '''Rewrites the corpus using the classes'''
  # read the corpus
  with open(txt_corpus) as f:
    corpus = []
    for sentence in f:
      corpus.append(np.array([cls_dict.get(w, 0) for w in sentence.strip().split()], int))
  # write it to disk
  with open(pickle_corpus, 'w') as fout:
    cPickle.dump(np.array([s for s in corpus], list), fout, protocol = cPickle.HIGHEST_PROTOCOL)

def load_txt_dict(txt_dict, pickle_dict):
  # read class dict
  cls_dict = defaultdict(int)
  with open(txt_dict) as f:
    for line in f:
      w, c = line.strip().split("\t")
      cls_dict[w] = c
  # write it to disk
  with open(pickle_dict, 'w') as f:
    cPickle.dump(cls_dict, f, protocol=cPickle.HIGHEST_PROTOCOL)
  return cls_dict

def load_pickle_dict(pickle_dict):
  with open(pickle_dict) as f:
    return cPickle.load(f)

if __name__ == '__main__':
  if len(sys.argv) == 3:
    load_txt_dict(sys.argv[1], sys.argv[2])
  elif len(sys.argv) == 4:
    cls = load_pickle_dict(sys.argv[2])
    txt_corpus, pickle_corpus = sys.argv[1], sys.argv[3]
    rewrite(txt_corpus, pickle_corpus, cls)
  else:
    sys.stderr.write('Usage: %s classes.txt classes.pickle\nOr %s corpus.txt classes.pickle corpus.pickle\n' % (sys.argv[0], sys.argv[0]))
    sys.exit(1)

