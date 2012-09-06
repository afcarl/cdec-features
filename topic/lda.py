import logging
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
        return self._lda[common.filter(words)]

    def topic_vector(self, words):
        return [v for k, v in self._lda.__getitem__(common.filter(words), eps=0)]

def main():
    logging.basicConfig(level=logging.INFO)
    lda = LDA('model.txt', 'vocab.txt', 'train.txt', topics = 10)
    lda.train()
    
    for sentence in ['i want to go', 'i want to leave']:
        for dist in lda.topics(sentence.split()):
            print dist

if __name__ == '__main__':
    main()
