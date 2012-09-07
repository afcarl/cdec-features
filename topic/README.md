# Topic similarity feature

1. Train the LDA model (here with 100 topics):

    python lda.py train.txt model.lda model.dic 100

2. Pre-compute topic distributions for the training corpus

    python store.py model.lda model.dic train.txt train-topics.pickle

3. Edit your suffix array extractor topic with the path of the trained models:

    lda_model = 'model.lda'
    lda_dict = 'model.dic'
    train_topics = 'train-topics.pickle'

4. Run the extractor with the feature
