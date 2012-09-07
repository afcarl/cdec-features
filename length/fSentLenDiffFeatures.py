import cdec.sa
import sys
import math

'''
Two length-based features for cdec.
Both features measure the difference between the length of the current source sentence and the average sentence length of the source training sentences which the current fphrase is extracted from. Thus, these features only access the source sentences of the training data.
The only difference between the two features lies in the computation of the "average sentence length".

@author Eva Mujdricza-Maydt (mujdricz@cl.uni-heidelberg.de)
'''
E = math.e

@cdec.sa.feature
def FSentLenDiff_arithm(ctx):
	'''
	This feature computes the difference between the length of the current source sentence and the average sentence length of the source training sentences which the current fphrase is extracted from. 
	The mentioned average sentence lenght is a kind of arithmetic average, and will be computed as follows:
	SUM_i=1..n(e**(-(|len_test-len_i|))) / n
	n: number of training sentences matching the fphrase
	len_test: length of the current source sentence to translate
	len_i: i-th training source sentence
	'''
	fsentLenList, testSentLen = extractFSentLens(ctx)
	return scoreFSentLenDiff_arithm(fsentLenList, testSentLen)

@cdec.sa.feature
def FSentLenDiff_median(ctx):
	'''
	This feature computes the absolute difference between the length of the current source sentence and the median of the sentence lengths among the source training sentences which the current fphrase is extracted from.
        '''
	fsentLenList, testSentLen = extractFSentLens(ctx)
	return scoreFSentLenDiff_median(fsentLenList, testSentLen)

def extractFSentLens(ctx):
	f = ctx.f_text
	testSent = cdec.sa.decode_sentence(ctx.test_sentence)
	testSentLen = len(testSent)-2 #minus 2 because of the "<s>" and "</s>" in the testSent-representation
	fsentLenList = [] #list of the lengths of the training sentences observed for fphrase
	for m in ctx.matches:
		fsentId = ctx.f_text.get_sentence_id(m[0]) #note that m should entail at least one entry (tuple) which all have the same sentence id
		fsent = ctx.f_text.get_sentence(fsentId)
		fsentLen = len(fsent) - 1 #minus 1 because of the 'END_OF_LINE'
		fsentLenList.append(fsentLen)
	
	return (fsentLenList, testSentLen)


def scoreSentLen(base, diff):
	return base ** -diff


def scoreFSentLenDiff_arithm(lengthList, testSentLen):
	numerator = math.fsum([scoreSentLen(E, (math.fabs(testSentLen-l))) for l in lengthList])
	return float(numerator) / len(lengthList)


def scoreFSentLenDiff_median(lengthList, testSentLen):
	lengthList.sort()
	h = len(lengthList)/2
	median = 0
	if h%2 == 0:
		median = math.fsum(lengthList[h-1:h+1])/2.0
	else:
		median = lengthList[h]

	return math.fabs(testSentLen - median)

