#  -*- Makefile -*-  for creating pyFormex releases
# $Id: Makefile 53 2005-12-05 18:23:28Z bverheg $
##
## This file is part of pyFormex 0.6 Release Fri Nov 16 22:39:28 2007
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
#

include RELEASE

PYFORMEXDIR= pyformex-${RELEASE}
PYSOURCE= ${addprefix pyformex/,${addsuffix .py, ${PYMODULES}}}
PYGUISOURCE= ${addprefix pyformex/gui/,${addsuffix .py,${PYGUIMODULES}}}
PLUGINSOURCE= ${addprefix pyformex/plugins/,${addsuffix .py,${PLUGINMODULES}}}
OTHERSOURCE= pyformex/pyformexrc
ICONFILES= $(wildcard icons/*.xpm) $(wildcard icons/pyformex_*.png)
DOCDIR= doc
HTMLDIR= ${DOCDIR}/html
HTMLDOCS= ${addprefix ${HTMLDIR}/,${PYSOURCE:.py=.html} }
HTMLGUIDOCS= ${addprefix ${HTMLDIR}/, ${addsuffix .html, gui ${addprefix gui.,${PYGUIMODULES}}}}
HTMLPLUGINDOCS= ${addprefix ${HTMLDIR}/, ${addsuffix .html, plugins ${addprefix plugins.,${PLUGINMODULES}}}}
STAMPFILES= README History Makefile FAQ
NONSTAMPFILES= COPYING RELEASE ReleaseNotes-${VERSION} Description 
EXAMPLEFILES= ${addprefix pyformex/examples/,${addsuffix .py, ${EXAMPLES} }}
IMAGEFILES=  ${addprefix screenshots/,${addsuffix .png,${IMAGES}}}

STAMPABLE= pyformex/${PROGRAM} ${PYSOURCE} ${OTHERSOURCE} ${PYGUISOURCE} ${PLUGINSOURCE}  ${EXAMPLEFILES} ${STAMPFILES}

STAMP= stamp 
VERSIONSTRING= __version__ = .*
NEWVERSIONSTRING= __version__ = "${RELEASE}"
PKG= ${PYFORMEXDIR}.tar.gz
PKGDIR= dist

.PHONY: dist pub distclean pydoc manual stamp dist.stamped version tag

############ Creating Distribution ##################


distclean:
	rm -rf ${PYFORMEXDIR}
	alldirs . "rm -f *~"

# Create the pydoc html files

pydoc: ${HTMLDIR}/index.html

${HTMLDIR}/index.html: ${HTMLDOCS} ${HTMLGUIDOCS} ${HTMLPLUGINDOCS} ${DOCDIR}/htmlindex.header ${DOCDIR}/htmlindex.footer
	./make_doc_index

${HTMLDIR}/%.html: %.py
	pydoc_gen.py $ -d ${HTMLDIR} $<

${HTMLDIR}/gui.%.html: gui/%.py
	pydoc_gen.py $ -d ${HTMLDIR} gui.$*

${HTMLDIR}/plugins.%.html: plugins/%.py
	pydoc_gen.py $ -d ${HTMLDIR} plugins.$*


# Create the manual
manual:
	make -C manual

# Create the C library
lib:
	make -C pyformex/lib

#pyformex/doc/pyformex-htmldocs.tar.gz: manual
#	tar czf $@ manual/html manual/images

# Set a new version

version: pyformex/globaldata.py manual/pyformex.tex setup.py

pyformex/globaldata.py: RELEASE
	sed -i 's|${VERSIONSTRING}|${NEWVERSIONSTRING}|' $@

manual/pyformex.tex: RELEASE
	sed -i 's|\\release{.*}|\\release{${RELEASE}}|;s|\\setshortversion{.*}|\\setshortversion{${VERSION}}|;'  $@


setup.py: RELEASE
	sed -i "s|version='.*'|version='${RELEASE}'|" $@


# Stamp files with the version/release date

stamp: Stamp.template RELEASE
	${STAMP} -tStamp.template version=${VERSION} -sStamp.stamp

stampall: stamp
	${STAMP} -tStamp.stamp -i ${STAMPABLE}

# Create the distribution
dist: ${PKGDIR}/${PKG}

${PKGDIR}/${PKG}: version #pyformex/doc/pyformex-htmldocs.tar.gz
	python setup.py sdist

# Publish the distribution to our ftp server
pub: 
	scp ${PKGDIR}/${PKG} bumps:/home/ftp/pub/pyformex


# Tag the release in the svn repository

tag:
	svn copy svn+ssh://svn.berlios.de/svnroot/repos/pyformex/trunk svn+ssh://svn.berlios.de/svnroot/repos/pyformex/tags/release-${RELEASE} -m "Tagging the ${RELEASE} release of the 'pyFormex' project."


minutes:
	rst2latex pyformex-user-meeting-0.rst > pyformex-user-meeting-0.tex
	pdflatex pyformex-user-meeting-0.tex
