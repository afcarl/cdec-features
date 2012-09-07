#!/usr/bin/env python
import sys
import logging
import cPickle
import numpy as np
import progressbar as pb
from lda import LDA

def main(model, dic, corpus, output):
    logging.basicConfig(level=logging.INFO)
    lda = LDA(model, dic)
    lda.load()
    topics = []
    with open(corpus) as fp:
        n_sentences = sum(1 for line in fp)
    logging.info('Computing topic vectors for %d sentences', n_sentences)
    bar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=n_sentences)
    with open(corpus) as fp:
        for sentence in bar(fp):
            topics.append(lda.topic_vector(sentence.split()))
    logging.info('Saving topic information to %s', output)
    with open(output, 'w') as fp:
        cPickle.dump(np.vstack(topics), fp, protocol=cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    if len(sys.argv) != 5:
        sys.stderr.write('Usage: %s model dict corpus.txt output.pickle\n' % sys.argv[0])
        sys.exit(1)
    main(*sys.argv[1:])
