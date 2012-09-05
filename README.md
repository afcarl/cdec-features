#Contextual features for cdec

To add a feature, write something like:

    import cdec.sa

    @cdec.sa.feature
    def my_feature(ctx):
        return - ctx.paircount + len(ctx.test_sentence)
