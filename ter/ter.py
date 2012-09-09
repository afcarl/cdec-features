import cdec.score
import cdec.sa
import math


@cdec.sa.feature
def AvgTER(ctx):
  ters = []
  test_sentence = ' '.join(cdec.sa.decode_sentence(ctx.test_sentence)[1:-1])
  for tup in ctx.matches:
    ref = ' '.join(ctx.f_text.get_sentence(ctx.f_text.get_sentence_id(tup[0]))[0:-1])
    ter = cdec.score.BLEU(ref).evaluate(test_sentence)
    ters.append(ter.score)
  return sum(ters)/len(ters)

