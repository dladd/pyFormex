#  -*- Makefile -*-  for installing pyFormex
# $Id$
#
############# SET THESE TO SUIT YOUR INSTALLATION ####################

# root of the installation tree: this is a reasonable default
prefix = ${DESTDIR}/usr/local
# where to install the executable files
bindir= ${prefix}/bin
# where to install pyformex modules: some may prefer to use ${PREFIX} 
libdir= ${prefix}/lib
# where to install the documentation
docdir= ${prefix}/share/doc

############# NOTHING CONFIGURABLE BELOW THIS LINE ###################
include RELEASE

PYFORMEXDIR= pyformex-${VERSION}
PYFORMEXREL= pyformex-${RELEASE}
INSTDIR= ${libdir}/${PYFORMEXDIR}
DOCINSTDIR= ${docdir}/${PYFORMEXDIR}
PROGRAM= pyformex
PYSOURCE= ${addsuffix .py, ${PYMODULES}}
SOURCE= ${PYSOURCE} pyformexrc
ICONS= icons/*.xbm
HTMLDIR= html
HTMLDOCS= ${addprefix ${HTMLDIR}/,${PYSOURCE:.py=.html}}
EXAMPLEFILES= ${addprefix examples/,${addsuffix .py, ${EXAMPLES} __init__}}
IMAGEFILES =  ${addprefix images/,${addsuffix .png,${IMAGES}}}
DOCFILES= README COPYING History Makefile FAQ

INSTALL= install -c
INSTALL_PROGRAM= ${INSTALL} -m 0755
INSTALL_DATA= ${INSTALL} -m 0644

.PHONY: install dist distclean

all:
	@echo "Do 'make install' to install pyformex"

############ User installation ######################

install: installdirs ${PROGRAM} ${SOURCE} ${ICONS} ${EXAMPLEFILES} ${DOCFILES} ${IMAGEFILES} ${HTMLDOCS}
	echo "config['docdir'] = '${DOCINSTDIR}'" >> pyformexrc
	${INSTALL_PROGRAM} ${PROGRAM} ${INSTDIR}
	${INSTALL_DATA} ${SOURCE} ${INSTDIR}
	${INSTALL_DATA} ${ICONS} ${INSTDIR}/icons
	${INSTALL_DATA} ${EXAMPLEFILES} ${INSTDIR}/examples
	${INSTALL_DATA} ${DOCFILES} ${DOCINSTDIR}
	${INSTALL_DATA} ${IMAGEFILES} ${DOCINSTDIR}/images
	${INSTALL_DATA} ${HTMLDOCS} ${DOCINSTDIR}/html
	ln -sfn ${INSTDIR}/${PROGRAM} ${bindir}/${PROGRAM}

installdirs:
	install -d ${bindir} ${INSTDIR} ${INSTDIR}/icons ${INSTDIR}/examples ${DOCINSTDIR} ${DOCINSTDIR}/images ${DOCINSTDIR}/html

uninstall:
	echo "There is no automatic uninstall procedure."""
	echo "Remove the entire pyformex directory from where you installed it."
	echo "Remove the symbolic link to the pyformex program."""
	echo "Remove the pyformex doc files."""

############ Creating Distribution ##################


${HTMLDIR}/%.html: %.py
	pydoc -w ./$< && mv $*.html ${HTMLDIR}

pydoc: ${HTMLDOCS}

dist: Makefile.dist pydoc
	${MAKE} -f Makefile.dist
