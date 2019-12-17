PAPER = circus-tent
TEX = $(wildcard *.tex)
FIGS = $(wildcard figures/*.pdf figures/*.png figures/*.jpg ./*.jpg)
FINAL = CircusTent-DeveloperGuide-1.0

.PHONY: all clean

$(PAPER).pdf: $(TEX) $(FIGS)
	echo $(FIGS)
	pdflatex $(PAPER)
	pdflatex $(PAPER)
	pdflatex $(PAPER)
	cp $(PAPER).pdf $(FINAL).pdf
clean:
	rm -f *.aux *.bbl *.blg *.log *.lof *.lol *.lot *.out *.toc $(PAPER).pdf $(FINAL).pdf
