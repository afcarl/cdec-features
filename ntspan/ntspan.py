import cdec.sa
from sys import stderr
import re
import pprint
from operator import mul
from math import log

@cdec.sa.feature
def NonTermCosine(ctx):   
   s_dict = dict()
   s_dict['fphrase'] = str(ctx.fphrase)
   s_dict['ephrase'] = str(ctx.ephrase)
   
   s_dict['test'] = dict()
   s_dict['test']['sentence'] = cdec.sa.decode_sentence(ctx.test_sentence)
   
   nt = getTestSpans(ctx.fphrase, ctx.input_span, s_dict['test']['sentence'])
   s_dict['test']['nterm_spans'] = nt

   s_dict['training'] = []
   
   counts1 = [dict() for i in range(len(nt))]
   counts2 = [dict() for i in range(len(nt))]
   
   for i, words in enumerate(s_dict['test']['nterm_spans']):
      for j, w in enumerate(words):
         if(words[0] == '<s>'):
            c = 1.0/2**(len(words) - j - 1)
         elif(words[-1] == '</s>'):
            c = 1.0/2**j
         else:
            c = 1.0
         counts1[i][w] = counts1[i].get(w, 0) + c
   s_dict['counts1'] = counts1
   
   for abs_terminal_spans in ctx.matches:
      t_dict = dict()
      t_m = abs_terminal_spans[0]
      t_dict['id'] = ctx.f_text.get_sentence_id(t_m)
      t_dict['sentence'] = ctx.f_text.get_sentence(t_dict['id'])
      t_dict['term_spans'] = getTrainTermSpans(ctx, abs_terminal_spans, t_dict)
      t_dict['nterm_spans'] = getTrainNontermSpans(ctx, t_dict)
      s_dict['training'].append(t_dict)
      
      for i, words in enumerate(t_dict['nterm_spans']):
         for j, w in enumerate(words):
            if(words[0] == '<s>'):
               c = 1.0/2**(len(words) - j - 1)
            elif(words[-1] == '</s>'):
               c = 1.0/2**j
            else:
               c = 1.0
            counts2[i][w] = counts2[i].get(w, 0) + c
            
   s_dict['counts2'] = counts2
   
   pairs = zip(s_dict['counts1'], s_dict['counts2'])
   #pprint.pprint(pairs, stderr)
   cosines = [cosine(*pair) for pair in pairs]
   
   if(cosines):
      geo_cosine = reduce(mul, cosines)**(1.0/len(cosines))
   else:
      geo_cosine = 1

   if(geo_cosine < 10**(-100)):
      return 100
   return -log(geo_cosine)

def cosine(counts1, counts2) :
   m1 = max(counts1.values())
   pcounts1 = dict([ (k, v/m1) for k,v in counts1.items() ])

   m2 = max(counts2.values())
   pcounts2 = dict([ (k, v/m2) for k,v in counts2.items() ])
   
   counter = sum([v * pcounts2.get(k,0) for k, v in pcounts1.items()])
   denom1 =  sum([v**2 for k, v in pcounts1.items()])**(0.5)
   denom2 =  sum([v**2 for k, v in pcounts2.items()])**(0.5)
   
   return counter/(denom1 + denom2)

def RuleToTokens(phrase):
   return str(phrase).split(' ')

def getTestSpans(fphrase, spans, sentence):
   tspans = []
   ntspans = []
   
   tokens = RuleToTokens(fphrase)
   
   span = [None, spans[0]-1]
   
   fphrase_reg = re.sub(r'\\\[X\\,\d\\\]', '(.*)', re.escape(str(fphrase)))
   sentence_joined = ' '.join(sentence)
   
   m = re.search(fphrase_reg, sentence_joined)
   if(m):
      for g in m.groups():
         span = g.split()
         ntspans.append(span)
   return ntspans

def getTrainNontermSpans(ctx, s_dict):
   spans = []
   matches = s_dict['term_spans'][:]
   i = 0
   
   left = None
   right = matches[0][0]-1
   
   tokens = RuleToTokens(ctx.fphrase)
   while(i < len(tokens)):
      if(re.match('\[X,(\d+)\]', tokens[i], re.IGNORECASE)):
         if(left == None):
            spans.append(['<s>'] + s_dict['sentence'][0:right+1])
         elif(right == None):
            sent = s_dict['sentence'][left:]
            if(sent[-1] == 'END_OF_LINE'):
               sent.pop()
            spans.append(sent + ['</s>'])
         else:
            spans.append(s_dict['sentence'][left:right+1])   
            
         i += 1
      else:
         m = matches.pop(0)
         if(not len(matches)):
            left = m[1] + 1
            right = None
         else:
            left = m[1] + 1
            right = matches[0][0] - 1
         i += m[1] - m[0] + 1
         
   return spans
   
def getTrainTermSpans(ctx, t, s_dict):
   (s_b, s_e) = (ctx.f_text.sent_index[s_dict['id']],
                  ctx.f_text.sent_index[s_dict['id'] + 1])
   lengths = []
   l = 0
   for s in RuleToTokens(ctx.fphrase):
      if(re.match('\[X,(\d+)\]', s, re.IGNORECASE)):
         if(l):
            lengths.append(l)
            l = 0
      else:
         l += 1
   if(l):
      lengths.append(l)
   return [(m - s_b, m - s_b + lengths[i] - 1) for (i, m) in enumerate(t)]
      
   