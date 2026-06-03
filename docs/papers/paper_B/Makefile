all: paper_B.pdf

paper_B.pdf: paper_B.tex
	pdflatex -interaction=nonstopmode paper_B.tex
	pdflatex -interaction=nonstopmode paper_B.tex

clean:
	rm -f *.aux *.log *.out *.pdf
