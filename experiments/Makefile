CDEC=~/tools/cdec/decoder/cdec
PRO=~/tools/cdec/pro-train/dist-pro.pl
SCORE=~/tools/cdec/mteval/fast_score
SAMPLE=./modulo.py
DATA=../../data

sa-de-en.ini:
	python -m cdec.sa.compile -b $(DATA)/train.de-en\
		-a $(DATA)/aligned.grow-diag-final-and -o sa-de-en -c $@

dev.de-en.txt:
	cat $(DATA)/newstest2009.de-en | python $(SAMPLE) 4 > $@
	sed -r "s/ \|\|\| .*//" $@ > dev.de.txt
	sed -r "s/.* \|\|\| //" $@ > dev.en.txt

test.de-en.txt:
	cat $(DATA)/newstest2010.de-en | python $(SAMPLE) 4 > $@
	sed -r "s/ \|\|\| .*//" $@ > test.de.txt
	sed -r "s/.* \|\|\| //" $@ > test.en.txt

dev.de.sgml: dev.de-en.txt sa-de-en.ini
	mkdir -p grammars
	python -m cdec.sa.extract -c sa-de-en.ini -g grammars/dev.de-en -j 12 > $@ < dev.de.txt

test.de.sgml: test.de-en.txt sa-de-en.ini
	mkdir -p grammars
	python -m cdec.sa.extract -c sa-de-en.ini -g grammars/test.de-en -j 12 > $@ < test.de.txt

# BLEU ~ 18 | 30m
tune: dev.de.sgml
	$(PRO) --ref-files dev.en.txt --source-file dev.de.sgml\
		--weights init-weights.ini --jobs 12 ../cdec.ini --max-iterations 10\
		2> protrain.log > /dev/null

# BLEU = 19.54 | 10m
test: test.de.sgml
	$(CDEC) -c ../cdec.ini -w protrain/weights.10 < test.de.sgml > hyp.en.txt
	$(SCORE) -r test.en.txt -i hyp.en.txt

clean:
	rm dev.de-en.txt dev.de.txt dev.en.txt\
		test.de-en.txt test.de.txt test.en.txt\
		dev.de.sgml test.de.sgml protrain.log
	rm -rf sa-de-en sa-de-en.ini protrain grammars 
