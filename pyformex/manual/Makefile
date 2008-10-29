# Makefile for the pyFormex manual
#
PKGNAME=pyformex

PYTHONDIR=/usr/lib/python2.4
MKHOWTO=mkhowto 
#${PYTHONDIR}/doc/tools/mkhowto
TEXINPUTS=${PYTHONDIR}/doc/texinputs
TGT=pyformex
SRC=${TGT}.tex pyformex.help 
HTMLDIR=html
FAVICON=../favicon.ico
OPTIONS= --up-title "pyFormex Home Page" --up-link "http://pyformex.berlios.de" --image-type png --favicon ${FAVICON} 
TESTOPTIONS= --l2h-init=fix.perl --keep
TESTOPTIONS= --keep

REMOVEFILES= images images.* img*.png index.dat *.pl WARNINGS pyfav.png *.how

MANDOCS= ../manual/html/*.html ../manual/html/*.css ../manual/html/*.png
MANIMAGES= ../manual/images/*.png
RSYNC= rsync -av -e ssh

# our local ftp server
FTPLOCAL=bumps:/home/ftp/pub/${PKGNAME}

# Default target
.PHONY all: pdf html

.PHONY ps: ${TGT}.ps

.PHONY pdf: ${TGT}.pdf

.PHONY html: html/${TGT}.html

pyformex.help: ../pyformex
	../pyformex --help > $@

${TGT}.ps: ${SRC}
	${MKHOWTO} --a4 --ps $<

${TGT}.pdf: ${SRC}
	${MKHOWTO} --a4 --pdf $<

${TGT}.ind: ${SRC}
	makeindex $<

html/${TGT}.html: ${SRC}
	rm -rf ${HTMLDIR}
	mkdir ${HTMLDIR}
	ln -s ../images ${HTMLDIR}/images
	${MKHOWTO} --html --dir ${HTMLDIR}  ${TESTOPTIONS} $(OPTIONS)  $<
	cd html; rm -f ${REMOVEFILES}

# Transfer the files to BerliOS
pub: #pdf html
	${RSYNC} pyformex.pdf bverheg@shell.berlios.de:/home/groups/ftp/pub/pyformex
	${RSYNC} ${MANDOCS} bverheg@shell.berlios.de:/home/groups/pyformex/htdocs/manual
	${RSYNC} ${MANIMAGES} bverheg@shell.berlios.de:/home/groups/pyformex/htdocs/images

# and to our ftp server
publocal:
	rsync -lt pyformex.pdf ${FTPLOCAL}

clean:
	rm *~ *.mpx