BOOK=fenics-tutorial-chinese

all:
	xelatex $(BOOK)

final:
	xelatex $(BOOK)
	biber $(BOOK)
	makeindex $(BOOK)
	xelatex $(BOOK)
	xelatex $(BOOK)

clean:
	rm -f *.aux *.mtc* *.out *.idx *.bcf *.log *~ *.ptc *.toc *.run.xml *.maf
