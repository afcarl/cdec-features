# Word-class similarity feature

1. Cluster the words using mkcls:

     mkcls -c80 -n10 -pcorpus.txt -Vmap.cls

2. Convert the word-class map using cPickle

     python store\_classes.py map.cls map.pickle

3. Annotate the training data and write it to disk using cPickle

     python store\_classes.py corpus.txt map.pickle corpus.pickle


3. Edit your config:

  cls\_dict = 'map.pickle'
  train\_cls = 'corpus.pickle'

4. Run the extractor with the feature
