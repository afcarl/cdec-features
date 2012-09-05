import sys
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from common import get_content

def main():
    lda = LdaModel.load('model.lda')
    dictionary = Dictionary.load('model.dic')
    for sentence in sys.stdin:
        doc = dictionary.doc2bow(get_content(sentence))
        print(lda[doc])


if __name__ == '__main__':
    main()
