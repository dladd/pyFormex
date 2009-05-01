# Makefile for the pyFormex manual
#
PKGNAME=pyformex
TGT=pyformex

MANSRC= pyformex.tex refman.tex pyformex.help
AUTOREF= array coords connectivity mydict odict olist script simple utils \
  colors draw image widgets \
  curve fe inertia isopar lima mesh \
  project properties section2d surface turtle units
  
AUTOSRC= $(addprefix ref-, $(addsuffix .tex, ${AUTOREF}))
SRC= ${MANSRC} ${AUTOSRC}

PYTHONDIR=/usr/lib/python2.5
MKHOWTO=mkhowto 
TEXINPUTS=${PYTHONDIR}/doc/texinputs

HTMLDIR=html
FAVICON=../favicon.ico
OPTIONS= --up-title "pyFormex Home Page" --up-link "http://pyformex.berlios.de" --image-type png --favicon ${FAVICON} 
TESTOPTIONS= --l2h-init=fix.perl --keep
TESTOPTIONS= --keep

REMOVEFILES= images images.* index.dat *.pl WARNINGS pyfav.png *.how

MANDOCS= ../manual/html/*.html ../manual/html/*.css ../manual/html/*.png
MANIMAGES= ../manual/images/*.png
RSYNC= rsync -av -e ssh

# our local ftp server
FTPLOCAL=bumps:/home/ftp/pub/${PKGNAME}


# recognized suffixes
.SUFFIXES: .tex .py .pdf

# where to search for source files
vpath %.py ../ ../gui/ ../plugins/

# Default target
.PHONY all: pdf html

.PHONY ps: ${TGT}.ps

.PHONY pdf: ${TGT}.pdf

.PHONY html: html/${TGT}.html

.PHONY: autosrc
autosrc: ${AUTOSRC}

# Create input
pyformex.help: ../pyformex
	../pyformex --help > $@

ref-%.tex: %.py py2tex.py
	py2tex.py $< > $@

# Create output
${TGT}.ps: ${MANSRC}
	${MKHOWTO} --a4 --ps $<

${TGT}.pdf: ${SRC}
	${MKHOWTO} --a4 --pdf $<

${TGT}.ind: ${MANSRC}
	makeindex $<

html/${TGT}.html: ${MANSRC}
	rm -rf ${HTMLDIR}
	mkdir ${HTMLDIR}
	ln -s ../images ${HTMLDIR}/images
	${MKHOWTO} --html --dir ${HTMLDIR}  ${TESTOPTIONS} $(OPTIONS)  $<
	cd html; rm -f ${REMOVEFILES}

# Transfer the files to BerliOS
pub: #pdf html
	${RSYNC} pyformex.pdf shell.berlios.de:/home/groups/ftp/pub/pyformex
	${RSYNC} ${MANDOCS} shell.berlios.de:/home/groups/pyformex/htdocs/manual
	${RSYNC} ${MANIMAGES} shell.berlios.de:/home/groups/pyformex/htdocs/images

# and to our ftp server
publocal:
	rsync -lt pyformex.pdf ${FTPLOCAL}

clean:
	rm *~ *.mpx