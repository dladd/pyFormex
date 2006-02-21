# Makefile for the pyFormex manual
#

PYTHONDIR=/usr/lib/python2.4
MKHOWTO=${PYTHONDIR}/doc/tools/mkhowto
TEXINPUTS=${PYTHONDIR}/doc/texinputs
TGT=manual
HTMLDIR=html
FAVICON=../../website/favicon.ico

# Default target
.PHONY all: pdf ps html

.PHONY pdf: ${TGT}.pdf

.PHONY ps: ${TGT}.ps

.PHONY html: html/${TGT}.html

${TGT}.pdf: ${TGT}.tex
	${MKHOWTO} --a4 --pdf $<

${TGT}.ps: ${TGT}.tex
	${MKHOWTO} --a4 --ps $<

html/${TGT}.html: ${TGT}.tex
	${MKHOWTO} --html --dir ${HTMLDIR} --image-type png  --favicon ${FAVICON} $<
