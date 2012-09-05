import logging
from gensim.corpora.textcorpus import TextCorpus
from gensim.models.ldamodel import LdaModel
from common import get_content

class SentenceDocCorpus(TextCorpus):
    def __init__(self, input):
        super(SentenceDocCorpus, self).__init__(input)

    def get_texts(self):
        with open(self.input) as f:
            for sentence in f:
                if not sentence.strip(): continue
                yield get_content(sentence)

def main():
    logging.basicConfig(level=logging.INFO)
    corpus = SentenceDocCorpus('/tmp/train.de')
    for doc in corpus:
        for id, _ in doc:
            pass
    lda = LdaModel(corpus, num_topics=400, id2word=corpus.dictionary)
    corpus.dictionary.save('model.dic')
    lda.save('model.lda')

if __name__ == '__main__':
    main()
