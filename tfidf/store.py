#!/usr/bin/env python
import sys
import logging
import cPickle
import progressbar as pb
import tfidf
from gensim.matutils import corpus2csc

def main(model, dic, corpus, output):
    logging.basicConfig(level=logging.INFO)
    score = tfidf.scorer(model, dic)
    transforms = []
    with open(corpus) as fp:
        n_sentences = sum(1 for line in fp)
    logging.info('Computing tf-idf vectors for %d sentences', n_sentences)
    bar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=n_sentences)
    with open(corpus) as fp:
        for sentence in bar(fp):
            transforms.append(score(sentence.split()))
    logging.info('Saving tf-idf information to %s', output)
    with open(output, 'w') as fp:
        cPickle.dump(corpus2csc(transforms), fp, protocol=cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    if len(sys.argv) != 5:
        sys.stderr.write('Usage: %s model dict corpus.txt output.pickle\n' % sys.argv[0])
        sys.exit(1)
    main(*sys.argv[1:])
