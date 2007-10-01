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
	${MKHOWTO} --a4 --pdf --keep $<

${TGT}.ind: ${SRC}
	makeindex $<

html/${TGT}.html: ${SRC}
	rm -rf ${HTMLDIR}
	${MKHOWTO} --html --dir ${HTMLDIR}  ${TESTOPTIONS} $(OPTIONS)  $<
