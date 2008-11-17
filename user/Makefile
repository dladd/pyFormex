# Makefile for the pyformex user meeting
#

SRC= $(wildcard pyformex-user-meeting-*.rst)
ALLPDF= ${SRC:.rst=.pdf}
PDF= $(wildcard pyformex-user-meeting-*.pdf)
RSYNC= rsync -av -e ssh
ACCOUNT= bverheg@shell.berlios.de

default:
	@echo "You should use an explicit target, e.g. make #"
	@echo "where # is the number of the meeting."

all: ${ALLPDF}


%: pyformex-user-meeting-%.pdf
	@echo "Done"

%.pdf: %.tex
	pdflatex $<

%.tex: %.rst
	rst2latex $< > $@


# Transfers the files to BerliOs
pub: 
	${RSYNC} ${PDF} ${ACCOUNT}:/home/groups/ftp/pub/pyformex


clean:
	rm -rf *.log *.aux *.out *.tex

mrproper: clean
	rm -rf *.pdf