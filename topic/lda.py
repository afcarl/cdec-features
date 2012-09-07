import sys
import logging
import numpy as np
from gensim.corpora.textcorpus import TextCorpus
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
import common

class SentenceDocCorpus(TextCorpus):
    def __init__(self, input):
        super(SentenceDocCorpus, self).__init__(input)

    def get_texts(self):
        with open(self.input) as f:
            for sentence in f:
                if not sentence.strip(): continue
                yield common.get_content(sentence)

class LDA(object):
    def __init__(self, model, vocab, corpus=None, topics=200, passes=1):
        self._model_file = model
        self._dict_file = vocab
        self._corpus_file = corpus
        self._topics = topics
        self._passes = passes

    def train(self):
        self._corpus = SentenceDocCorpus(self._corpus_file)
        self._lda = LdaModel(self._corpus, num_topics = self._topics, id2word = self._corpus.dictionary, passes = self._passes)
        self._dictionary = self._corpus.dictionary
        
        self._lda.save(self._model_file)
        self._dictionary.save(self._dict_file)

    def load(self):
        self._lda = LdaModel.load(self._model_file)
        self._dictionary = Dictionary.load(self._dict_file)

    def topics(self, words):
        return self._lda[self._dictionary.doc2bow(common.filter(words))]

    def topic_vector(self, words):
        return np.array([v for k, v in self._lda.__getitem__(self._dictionary.doc2bow(common.filter(words)), eps=0)])

def main(train, model, dic, topics):
    logging.basicConfig(level=logging.INFO)
    lda = LDA(model, dic, train, topics=int(topics))
    lda.train()

if __name__ == '__main__':
    if len(sys.argv) != 5:
        sys.stderr.write('Usage: %s train.txt model dict topics\n' % sys.argv[0])
        sys.exit(1)
    main(*sys.argv[1:])
