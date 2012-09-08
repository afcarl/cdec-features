# Topic similarity feature

1. Cluster the words using mkcls:

    

2. Convert the word-class map using cPickle

  python store\_classes.py map.txt map.pickle

3. Annotate the training data and write it to disk using cPickle

  python store\_classes.py train.txt dict.pickle train.pickle


3. Edit your config:

  cls\_dict = 'data/dict.pickle'
  cls\_train = 'data/train.pickle'

4. Run the extractor with the feature
