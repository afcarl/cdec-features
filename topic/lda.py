import logging
from gensim.corpora.textcorpus import TextCorpus
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from common import get_content
import sys

class SentenceDocCorpus(TextCorpus):
    def __init__(self, input):
        super(SentenceDocCorpus, self).__init__(input)

    def get_texts(self):
        with open(self.input) as f:
            for sentence in f:
                if not sentence.strip(): continue
                yield get_content(sentence)

class LDA(object):
  
  def __init__(self, corpus, model, vocab, topics = 200, passes = 1):
    self._corpus_file = corpus
    self._model_file = model
    self._dict_file = vocab
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

  def clean_and_score(self, docs):
    '''This calls get_content'''
    results = []
    for sentence in docs:
      print sentence
      doc = self._dictionary.doc2bow(get_content(sentence))
      results.append(self._lda[doc])
    return results

  def score(self, docs):
    '''This does not call get_content'''
    results = []
    for sentence in docs:
      print sentence
      doc = self._dictionary.doc2bow(sentence)
      results.append(self._lda[doc])
    return results



def main():
  training = False
  logging.basicConfig(level=logging.INFO)
  lda = LDA('train.txt', 'model.txt', 'vocab.txt', topics = 10)
  if training:
    lda.train()
  else:
    lda.load()

  for dist in lda.clean_and_score(['i want to go', 'i want to leave']):
    print dist

if __name__ == '__main__':
    main()
