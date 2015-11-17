# receipe: http://www.qscitech.info/blog-entries/LaTeX-in-your-figures-with-XFig-shudders-.html

xfig -specialtext -latexfonts -startlatexFont default layers.fig
# write latex text as straight default text with $...$
# export to combined PS/LaTeX format

# copy tex file (like layers.tex) to some file, go into it and inset the right
# filename (layers.pstex_t)
latex layers.tex
# make EPS figure out of the xfig figure:
dvips -E layers.dvi -o layers.eps
# check layers.eps - you probably need to adjust the placement of $...$ text
# include layers.eps in the latex document



