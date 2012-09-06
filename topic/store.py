#!/usr/bin/env python
import sys
import logging
import cPickle
from lda import LDA

def main(model, dic, corpus, output):
    logging.basicConfig(level=logging.INFO)
    lda = LDA(model, dic)
    lda.load()
    topics = []
    with open(corpus) as fp:
        for sentence in fp:
            topics.append(lda.topic_vector(sentence.split()))
    with open(output, 'w') as fp:
        cPickle.dump(topics, fp, protocol=cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    if len(sys.argv) != 5:
        sys.stderr.write('Usage: %s model dict corpus.txt output.pickle\n' % sys.argv[0])
        sys.exit(1)
    main(*sys.argv[1:])
