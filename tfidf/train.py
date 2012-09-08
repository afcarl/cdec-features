import sys
import logging
from gensim.corpora.textcorpus import TextCorpus
from gensim.models import TfidfModel

class SentenceDocCorpus(TextCorpus):
    def get_texts(self):
        with open(self.input) as f:
            for sentence in f:
                yield sentence.split()

def main(train, model, dic):
    logging.basicConfig(level=logging.INFO)
    corpus = SentenceDocCorpus(train)
    tfidf = TfidfModel(corpus)
    tfidf.save(model)
    corpus.dictionary.save(dic)
    
if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.stderr.write('Usage: %s train.txt model dict\n' % sys.argv[0])
        sys.exit(1)
    main(*sys.argv[1:])
