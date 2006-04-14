#  -*- Makefile -*-  for installing pyFormex
# $Id$
#
############# SET THESE TO SUIT YOUR INSTALLATION ####################

# !! An environment variable DESTDIR can be set to specify a path for
# !! the installation tree. All install paths specified hereafter will
# !! be relative to that installation path.

# root of the installation tree
prefix = /usr/local
# where to install pyformex files: some may prefer to use ${PREFIX} 
libdir= ${prefix}/lib
# where to create a link to the executable files
bindir= ${prefix}/bin
# where to create a link to the documentation
docdir= ${prefix}/share/doc

############# NOTHING CONFIGURABLE BELOW THIS LINE ###################
include RELEASE

PYFORMEXVER= pyformex-${VERSION}
PYFORMEXREL= pyformex-${RELEASE}
INSTDIR= ${libdir}/${PYFORMEXVER}
DOCINSTDIR= ${libdir}/${PYFORMEXVER}/doc
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

.PHONY: install dist distclean manual

all:
	@echo "Do 'make install' to install pyformex"

############ User installation ######################

install: installdirs ${PROGRAM} ${SOURCE} ${ICONS} ${EXAMPLEFILES} ${DOCFILES} ${IMAGEFILES} ${HTMLDOCS}
	${INSTALL_PROGRAM} ${PROGRAM} ${DESTDIR}${INSTDIR}
	${INSTALL_DATA} ${SOURCE} ${DESTDIR}${INSTDIR}
	${INSTALL_DATA} ${ICONS} ${DESTDIR}${INSTDIR}/icons
	${INSTALL_DATA} ${EXAMPLEFILES} ${DESTDIR}${INSTDIR}/examples
	${INSTALL_DATA} ${DOCFILES} ${DESTDIR}${DOCINSTDIR}
	${INSTALL_DATA} ${IMAGEFILES} ${DESTDIR}${DOCINSTDIR}/images
	${INSTALL_DATA} ${HTMLDOCS} ${DESTDIR}${DOCINSTDIR}/html
	${call makesymlink,${PROGRAM},${PYFORMEXVER}/${PROGRAM}}
	ln -sfn ${DOCINSTDIR} ${DESTDIR}${docdir}/${PYFORMEXVER}

# create a symlink $(1) in $(bindir) pointing to $(2) in $(libdir)
# this will detect the special cases where $(bindir)==$(libdir)/bin or
# $(bindir)==$(libdir)/../bin, and make a short relative symlink.
makesymlink= if [ $(bindir) = $(subst lib,bin,$(libdir)) ]; then ln -sfn ../lib/$(2) ${DESTDIR}$(bindir)/$(1); elif [ "$(bindir)" = "$(libdir)/bin" ]; then ln -sfn ../$(2) ${DESTDIR}$(bindir)/$(1); else ln -sfn $(libdir)/$(2) ${DESTDIR}$(bindir)/$(1); fi

installdirs:
	install -d ${DESTDIR}${bindir} ${DESTDIR}${docdir} ${DESTDIR}${INSTDIR} ${DESTDIR}${INSTDIR}/icons ${DESTDIR}${INSTDIR}/examples ${DESTDIR}${DOCINSTDIR} ${DESTDIR}${DOCINSTDIR}/images ${DESTDIR}${DOCINSTDIR}/html

uninstall:
	echo "There is no automatic uninstall procedure."""
	echo "Remove the entire pyformex directory from where you installed it."
	echo "Remove the symbolic link to the pyformex program."""
	echo "Remove the pyformex doc files."""

${HTMLDIR}/%.html: %.py
	pydoc -w ./$< && mv $*.html ${HTMLDIR}

clean:
	rm -f *.pyc
	rm -f examples/*.pyc

################# SHORTHANDS FOR DEVELOPERS ONLY ##################

dist: Makefile.dist
	${MAKE} -f Makefile.dist

version: Makefile.dist
	${MAKE} -f Makefile.dist version

manual: Makefile.dist
	${MAKE} -f Makefile.dist manual
