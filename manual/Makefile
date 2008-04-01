# Makefile for the pyFormex manual
#

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

# Default target
.PHONY all: pdf html

.PHONY ps: ${TGT}.ps

.PHONY pdf: ${TGT}.pdf

.PHONY html: html/${TGT}.html

pyformex.help: ../pyformex/pyformex
	../pyformex/pyformex --help > $@

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

# Transfers the files to BerliOS
pub: pdf
	scp pyformex.pdf bverheg@shell.berlios.de:/home/groups/ftp/pub/pyformex

