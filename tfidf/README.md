# TF-IDF feature

1. Train the model:

    python train.py train.txt model dict

2. Pre-compute for the training corpus

    python store.py model dict corpus.txt corpus.pickle

3. Edit your suffix array extractor topic with the path of the trained models:

    tfidf_model = 'model'
    tfidf_dict = 'dict'
    train_tfidf = 'corpus.pickle'

4. Run the extractor with the feature
