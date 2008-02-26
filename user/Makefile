# Makefile for the pyformex user meeting
#

SRC= pyformex-user-meeting-*.rst
PDF= pyformex-user-meeting-*.pdf
RSYNC= rsync -av -e ssh
ACCOUNT= bverheg@shell.berlios.de

minutes: pyformex-user-meeting-0.tex
	pdflatex pyformex-user-meeting-0.tex

pyformex-user-meeting-0.tex: pyformex-user-meeting-0.rst
	rst2latex pyformex-user-meeting-0.rst > pyformex-user-meeting-0.tex


# Transfers the files to BerliOs
pub: 
	$(RSYNC) $(PDF) $(ACCOUNT):/home/groups/ftp/pub/pyformex
#	scp filename.tar.gz yourusername@shell.berlios.de:/home/groups/ftp/pub/yourprojectname