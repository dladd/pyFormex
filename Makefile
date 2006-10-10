#  -*- Makefile -*-  for installing pyFormex
# $Id$
#
############# SET THESE TO SUIT YOUR INSTALLATION ####################

# !! An environment variable DESTDIR can be set to specify a path for
# !! the installation tree. All install paths specified hereafter will
# !! be relative to that installation path.

# root of the installation tree
prefix = /usr/local
# where to install pyformex files: some may prefer to use ${prefix} or
# ${prefix}/share 
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
PROGLINK= pyformex
PYSOURCE= ${addsuffix .py, ${PYMODULES}}
PYGUISOURCE= ${addprefix gui/,${addsuffix .py,${PYGUIMODULES}}}
OTHERSOURCE= pyformexrc
ICONFILES= icons/*.xpm
HTMLDIR= html
HTMLDOCS= ${addprefix ${HTMLDIR}/,${PYSOURCE:.py=.html}} ${HTMLDIR}/index.html
EXAMPLEFILES= ${addprefix examples/,${addsuffix .py, ${EXAMPLES} }}
IMAGEFILES =  ${addprefix images/,${addsuffix .png,${IMAGES}}}
DOCFILES= README COPYING History Makefile FAQ Description

INSTALL= install -c
INSTALL_PROGRAM= ${INSTALL} -m 0755
INSTALL_DATA= ${INSTALL} -m 0644

.PHONY: install dist distclean manual

all:
	@echo "Do 'make install' to install pyformex"

############ User installation ######################

install: installdirs ${SOURCE} ${ICONS} ${EXAMPLEFILES} ${DOCFILES} ${IMAGEFILES} ${HTMLDOCS}
	${INSTALL_PROGRAM} ${PROGRAM} ${DESTDIR}${INSTDIR}
	${INSTALL_DATA} ${PYSOURCE} ${OTHERSOURCE} ${DESTDIR}${INSTDIR}
	${INSTALL_DATA} ${PYGUISOURCE} ${DESTDIR}${INSTDIR}/gui
	${INSTALL_DATA} ${ICONFILES} ${DESTDIR}${INSTDIR}/icons
	${INSTALL_DATA} ${EXAMPLEFILES} ${DESTDIR}${INSTDIR}/examples
	${INSTALL_DATA} ${DOCFILES} ${DESTDIR}${DOCINSTDIR}
	${INSTALL_DATA} ${IMAGEFILES} ${DESTDIR}${DOCINSTDIR}/images
	${INSTALL_DATA} ${HTMLDOCS} ${DESTDIR}${DOCINSTDIR}/html
	${call makesymlink,${PROGLINK},${PYFORMEXVER}/${PROGRAM}}
	ln -sfn ${DOCINSTDIR} ${DESTDIR}${docdir}/${PYFORMEXVER}
	${INSTALL_DATA} ${MANUAL} ${DESTDIR}${INSTDIR}/manual
	${INSTALL_DATA} ${MANUALHTML} ${DESTDIR}${INSTDIR}/manual/html
	${INSTALL_DATA} ${MANUALIMAGES} ${DESTDIR}${INSTDIR}/manual/images
	[ -z "$$(type -P xdg-desktop-menu)" ] || xdg-desktop-menu install ${VENDOR}-${DESKTOPFILE}

# create a symlink $(1) in $(bindir) pointing to $(2) in $(libdir)
# this will detect the special cases where $(bindir)==$(libdir)/bin or
# $(bindir)==$(libdir)/../bin, and make a short relative symlink.
makesymlink= if [ $(bindir) = $(subst lib,bin,$(libdir)) ]; then ln -sfn ../lib/$(2) ${DESTDIR}$(bindir)/$(1); elif [ "$(bindir)" = "$(libdir)/bin" ]; then ln -sfn ../$(2) ${DESTDIR}$(bindir)/$(1); else ln -sfn $(libdir)/$(2) ${DESTDIR}$(bindir)/$(1); fi

installdirs:
	install -d ${DESTDIR}${bindir} ${DESTDIR}${docdir} ${DESTDIR}${INSTDIR} ${DESTDIR}${INSTDIR}/gui ${DESTDIR}${INSTDIR}/icons ${DESTDIR}${INSTDIR}/examples ${DESTDIR}${DOCINSTDIR} ${DESTDIR}${DOCINSTDIR}/images ${DESTDIR}${DOCINSTDIR}/html ${DESTDIR}${INSTDIR}/manual ${DESTDIR}${INSTDIR}/manual/html ${DESTDIR}${INSTDIR}/manual/images

uninstall:
	echo "There is no automatic uninstall procedure yet."""
	echo "Remove the entire pyformex directory from where you installed it."
	echo "Remove the symbolic link to the pyformex program."""

${HTMLDIR}/%.html: %.py
	pydoc -w ./$< && mv $*.html ${HTMLDIR}

clean:
	rm -f *.pyc
	rm -f examples/*.pyc

################# SHORTHANDS FOR DEVELOPERS ONLY ##################

dist: Makefile.dist
	${MAKE} -f Makefile.dist dist

pub: Makefile.dist
	${MAKE} -f Makefile.dist pub

version: Makefile.dist
	${MAKE} -f Makefile.dist version

manual: Makefile.dist
	${MAKE} -f Makefile.dist manual
