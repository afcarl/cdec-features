#Contextual features for cdec

To add a feature, write something like:

```python
import cdec.sa

@cdec.sa.configure
def configure(config):
    pass

@cdec.sa.annotator
def my_annotation(words):
    return None

@cdec.sa.feature
def my_feature(ctx):
    return - ctx.paircount + len(ctx.test_sentence)
```
