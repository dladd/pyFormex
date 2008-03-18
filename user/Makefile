# Makefile for the pyformex user meeting
#

SRC= pyformex-user-meeting-*.rst
PDF= ${SRC:.rst=.pdf}
RSYNC= rsync -av -e ssh
ACCOUNT= bverheg@shell.berlios.de

minutes: ${PDF}

%.pdf: %.tex
	pdflatex $<

%.tex: %.rst
	rst2latex $< > $@


# Transfers the files to BerliOs
pub: 
	$(RSYNC) $(PDF) $(ACCOUNT):/home/groups/ftp/pub/pyformex
#	scp filename.tar.gz yourusername@shell.berlios.de:/home/groups/ftp/pub/yourprojectname