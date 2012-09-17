import cdec.sa
from math import log10
import re
import sys
from itertools import izip


test_sent_id = -1

#@cdec.sa.config
def configure(config):
# get names of file with train and test set pos tag information
	train_tag_file = config['train_tag']
	test_tag_file = config['test_tag']

train_tag_file = "/home/gesa/cl/mt/mtm/fbis/corpus.puretag.zh"
test_tag_file = "/home/gesa/cl/mt/mtm/fbis/dev_and_test/mt03.puretag.zh"
# load train and test set tagging into some data structure
train_tags = cdec.sa.DataArray(from_text=train_tag_file, use_sent_id=True)
test_tags = cdec.sa.DataArray(from_text=test_tag_file, use_sent_id=True)


@cdec.sa.annotator
def pos_tags(sentence):
	global test_sent_id 
	test_sent_id += 1
	return test_tags.get_sentence(test_sent_id)
	

@cdec.sa.feature
def nt_boundary_pos(ctx):
	global test_sent_id	
	src_terminals = [s.split() for s in filter(None,re.split("\[X,[12]\]", str(ctx.fphrase)))]
	num_nts = len(re.findall("\[X,[12]\]", str(ctx.fphrase)))
	scores = []
	if num_nts > 0:
#		sys.stderr.write("FPHRASE: %s\n"%str(ctx.fphrase))
#		sys.stderr.write("TEST SENTENCE: %s\n"%" ".join(cdec.sa.decode_sentence(ctx.test_sentence)))
		test_nt_spans = get_nt_spans(ctx, num_nts, src_terminals)
#		sys.stderr.write("TEST_NT_SPANS: %s\n"%str(test_nt_spans))
#		sys.stderr.write("\n")
		test_nt_span_pos_tags = get_pos_tags(ctx, test_nt_spans)
		nt_span_dist = {}
		for i in range(len(test_nt_span_pos_tags)):
			span = test_nt_span_pos_tags[i]
			nt_span_dist["left%d"%i] = {}
			nt_span_dist["right%d"%i] = {}
#			sys.stderr.write("SPAN: %s\n"%str(span))
#			sys.stderr.write("BOUNDARIES: %s %s\n"%(str(span[0]), str(span[-1])))
#		sys.stderr.write("\n")
#		sys.stderr.write("NUMBER OF MATCHES IN TRAIN: %d\n"%len(ctx.matches))
		for match in ctx.matches:
			train_sent_id = ctx.f_text.get_sentence_id(match[0])
			train_nt_spans = get_nt_spans(ctx, num_nts, src_terminals, train_sent_id)
			train_nt_span_pos_tags = get_pos_tags(ctx, train_nt_spans, train_sent_id)
			for i in range(len(train_nt_span_pos_tags)):
				span = train_nt_span_pos_tags[i]
#				sys.stderr.write("SPANS: %s %s\n"%(span[0], span[-1])) 
				nt_span_dist["left%d"%i][span[0]] = nt_span_dist["left%d"%i].get(span[0], 0) + 1
				nt_span_dist["right%d"%i][span[-1]] = nt_span_dist["right%d"%i].get(span[-1], 0) + 1
		marginal = lambda x: sum(x.itervalues())
		for i in range(len(test_nt_span_pos_tags)):
			span = test_nt_span_pos_tags[i]
			if not span[0] == "BOS":
#				sys.stderr.write("DIST LEFT: %s\n"%str(nt_span_dist["left%d"%i]))
				scores.append(1.0*nt_span_dist["left%d"%i].get(span[0], 0)/marginal(nt_span_dist["left%d"%i]))
			if not span[-1] == "EOS":
#				sys.stderr.write("DIST RIGHT: %s\n"%str(nt_span_dist["right%d"%i]))
				scores.append(1.0*nt_span_dist["right%d"%i].get(span[-1], 0)/marginal(nt_span_dist["right%d"%i]))
	if scores:
		score = 1.0*sum(scores)/len(scores)
#		sys.stderr.write("SCORE: %d\n"%score)
#		sys.stderr.write("\n")
#		sys.stderr.write("SCORE: %f\n"%score)
		if score == 0:
			score = 0.1
	else:
		score = 0.1
	return -log10(score)

def get_nt_spans(ctx, num_nts, src_terminals, sent_id=-1):
#	sys.stderr.write("SENT_ID: %d\n"%sent_id)
	fphrase = str(ctx.fphrase)
	regex = "(" + re.sub(r"\\\[X\\\,[12]\\\]", "(.*)", re.escape(fphrase)) + ")"
	if sent_id == -1:
		# get nt_spans for test sentence
		sent = cdec.sa.decode_sentence(ctx.test_sentence)
#		sys.stderr.write("REGEX: %s\n"%regex)
	else: 
		# remove EOL and add <s> and </s> instead
		sent = ["<s>"] + ctx.f_text.get_sentence(sent_id)[0:-1] + ["</s>"]
#	sys.stderr.write("SENTENCE: %s\n"%(" ".join(sent)))
#	sys.stderr.write("SENTENCE: %s\n"%(" ".join(sent)))
	matched_groups = re.search(regex, " ".join(sent)).groups()
	matched, nt_matches = matched_groups[0], matched_groups[1:]
#	for match in matched_groups:
#		sys.stderr.write("\t%s\n"%match)
#	sys.stderr.write("MATCHED: %s\n"%matched)
#	for stuff in [s.split() for s in filter(None, re.split(re.escape(matched), "NOTHING " + " ".join(sent) + " NOTHING"))]:
#		sys.stderr.write("STUFF BEFORE/AFTER: %s\n"%stuff)
	tokens_before_match, tokens_after_match = [s.split() for s in filter(None, re.split(re.escape(matched), "NOTHING " + " ".join(sent) + " NOTHING"))]
	tokens_before_match.remove("NOTHING")
	tokens_after_match.remove("NOTHING")
#	sys.stderr.write("SENTENCE: %s\n"%(" ".join(sent)))
#	sys.stderr.write(str(type(tokens_before_match)))
#	sys.stderr.write("STUFF BEFORE: %s\n"%(" ".join(tokens_before_match if tokens_before_match else [])))
#	sys.stderr.write("STUFF AFTER: %s\n"%(" ".join(tokens_after_match if tokens_after_match else [])))
	num_tokens_before_match, num_tokens_after_match = [(0 if not t else len(t)) for t in (tokens_before_match, tokens_after_match)]
	left_unbound = re.match("^\[X,[12]\].*", fphrase)
	right_unbound = re.match(".*\[X,[12]\]$", fphrase)
	nt_spans = []
	indices_nt1 = []
	indices_nt2 = []
	if num_nts == 2:
		nt1, nt2 = [s.split() for s in nt_matches]
		if len(src_terminals) == 3:
			index_nt1_left = num_tokens_before_match + len(src_terminals[0])
			index_nt1_right = num_tokens_before_match + len(src_terminals[0]) + len(nt1)
			index_nt2_left = len(sent) - num_tokens_after_match - len(src_terminals[2]) - len(nt2)
			index_nt2_right = len(sent) - num_tokens_after_match - len(src_terminals[2])
		else:
			if left_unbound:
				# re.search will greedily match everything from BOS until first terminal as (.*)
				index_nt1_left = 0
				index_nt1_right = len(nt1)
				if len(src_terminals) == 2:
					index_nt2_left = len(sent) - num_tokens_after_match - len(src_terminals[1]) - len(nt2)
					index_nt2_right = len(sent) - num_tokens_after_match - len(src_terminals[1])
			if right_unbound:
				if len(src_terminals) == 2:
					index_nt1_left = num_tokens_before_match + len(src_terminals[0])
					index_nt1_right = num_tokens_before_match + len(src_terminals[0]) + len(nt1)
				# re.search will greedily match everything from last terminal until EOS as (.*)
				index_nt2_left = len(sent) - len(nt2)
				index_nt2_right = len(sent)		
		indices_nt1.extend([index_nt1_left,index_nt1_right])
		indices_nt2.extend([index_nt2_left, index_nt2_right])
#		sys.stderr.write("LEN NT2: %d\n"%len(nt2))
	else:
		if num_nts == 1:
			nt1 = nt_matches[0].split()
#			sys.stderr.write("%s\n"%str(nt1))
#			sys.stderr.write("%d\n"%len(nt1))
			if left_unbound:
				# re.search will greedily match everything from BOS until first terminal as (.*)
				index_nt1_left = 0
				index_nt1_right = len(nt1)
			else:
				if right_unbound:
					# re.search will greedily match everything from last terminal until EOS as (.*)
					index_nt1_left = len(sent) - len(nt1)
					index_nt1_right = len(sent)
				else:
					index_nt1_left = num_tokens_before_match + len(src_terminals[0])
					index_nt1_right = num_tokens_before_match + len(src_terminals[0]) + len(nt1)
			indices_nt1.extend((index_nt1_left, index_nt1_right))
	nt_spans.append(indices_nt1)
	nt_spans.append(indices_nt2)
	return nt_spans
				
def get_pos_tags(ctx, spans, sent_id=-1):
	return_pos_tags = []
	if sent_id == -1:
		# get POS tags for test sentence
		sent_postags = ["BOS"] + ctx.meta['pos_tags'][0:-1] + ["EOS"]
	else:
		sent_postags = ["BOS"] + train_tags.get_sentence(sent_id)[0:-1] + ["EOS"]
#	sys.stderr.write("LEN SENT_POSTAGS: %d\n"%len(sent_postags))
#	sys.stderr.write("%s\n"%str(sent_postags))
#	sys.stderr.write("POS SPANS WANTED: %s\n"%str(spans))
	for span in spans:
		if span:
			return_pos_tags.append(sent_postags[span[0]:span[1]])
#			if sent_id > -1:
#				sys.stderr.write("%s\n"%str(sent_postags[span[0]:span[1]]))
	return return_pos_tags
