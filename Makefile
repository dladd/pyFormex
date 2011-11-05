#  -*- Makefile -*-  for creating pyFormex releases
# $Id$
##
##  This file is part of pyFormex 0.8.4 Release Sat Jul  9 14:43:11 2011
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Homepage: http://pyformex.org   (http://pyformex.berlios.de)
##  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##  Distributed under the GNU General Public License version 3 or later.
##
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see http://www.gnu.org/licenses/.
##
#

include RELEASE

PKGNAME= pyformex

PYFORMEXDIR= pyformex

LIBDIR= ${PYFORMEXDIR}/lib
DOCDIR= ${PYFORMEXDIR}/doc
BINDIR= ${PYFORMEXDIR}/bin
SPHINXDIR= sphinx

SOURCE= ${PYFORMEXDIR}/pyformex \
	$(wildcard ${PYFORMEXDIR}/*.py) \
	$(wildcard ${PYFORMEXDIR}/gui/*.py) \
	$(wildcard ${PYFORMEXDIR}/plugins/*.py) \
	$(wildcard ${LIBDIR}/*.c) $(wildcard ${LIBDIR}/*.py) \
	$(addprefix ${BINDIR}/, read_abq_inp.awk) \


EXAMPLES= \
	$(wildcard ${PYFORMEXDIR}/examples/*.py) \
	$(wildcard ${PYFORMEXDIR}/examples/Analysis/*.py) \
	$(wildcard ${PYFORMEXDIR}/examples/Demos/*.py) \

EXAMPLEDATA= $(wildcard ${PYFORMEXDIR}/examples/*.db)

DOCSOURCE= $(wildcard ${SPHINXDIR}/*.rst)

EXECUTABLE= ${PYFORMEXDIR}/pyformex ${PYFORMEXDIR}/sendmail.py ${BINDIR}/read_abq_inp.awk ${LIBDIR}/postabq pyformex-viewer


OTHERSTAMPABLE= setup.py Makefile post-install \
	${PYFORMEXDIR}/pyformexrc \
	${EXAMPLEDATA} \
	${LIBDIR}/Makefile.in \
	${addprefix ${DOCDIR}/, README ReleaseNotes} \

NONSTAMPABLE= ${DOC}/COPYING 

STAMPABLE= $(filter-out ${PYFORMEXDIR}/template.py,${SOURCE}) \
	${EXAMPLES} ${DOCSOURCE} ${OTHERSTAMPABLE}


STAMP= stamp 
VERSIONSTRING= __version__ = .*
NEWVERSIONSTRING= __version__ = "${RELEASE}"

PKGVER= ${PKGNAME}-${RELEASE}.tar.gz
PKGDIR= dist
LATEST= ${PKGNAME}-latest.tar.gz

# our local ftp server
FTPLOCAL=bumps:/var/ftp/pub/pyformex
# ftp server on pyformex website
FTPPYFORMEX=bverheg@shell.berlios.de:/home/groups/ftp/pub/pyformex

.PHONY: dist pub distclean html pdf pubdoc minutes website stamp dist.stamped version tag register bumprelease bumpversion

############ Creating Distribution ##################

default:
	@echo Please specify a target

distclean:
	alldirs . "rm -f *~"

# Create the C library
lib: ${LIBDIR}/Makefile
	make -C ${LIBDIR}

# Create the C library with debug option
libdebug: ${LIBDIR}/Makefile
	make -C ${LIBDIR} debug

# Create the C library without debug option
libnodebug: ${LIBDIR}/Makefile
	make -C ${LIBDIR} nodebug

# Clean C library
libreset: ${LIBDIR}/Makefile
	make -C ${LIBDIR} reset

${LIBDIR}/Makefile: ${LIBDIR}/configure
	cd ${LIBDIR} && ./configure

# Create the minutes of the user meeting
minutes: 
	make -C user

# Create the website
website: 
	make -C website


# Bump the version/release
bumpversion:
	OLD=$$(expr "${VERSION}" : '.*\([0-9])*\)$$');NEW=$$(expr $$OLD + 1);sed -i "/^VERSION=/s|$$OLD$$|$$NEW|" RELEASE
	sed -i '/^RELEASE=/s|}.*|}-a1|' RELEASE
	make version

# This increases the tail only: minor number or alpha number
bumprelease:
	OLD=$$(expr "${RELEASE}" : '.*\([0-9])*\)$$');NEW=$$(expr $$OLD + 1);sed -i "/^RELEASE=/s|$$OLD$$|$$NEW|" RELEASE
	make version

revision:
	sed -i "s|__revision__ = .*|__revision__ = '$$(svnversion)'|" ${PYFORMEXDIR}/__init__.py

version: ${PYFORMEXDIR}/__init__.py setup.py ${LIBDIR}/configure.ac ${SPHINXDIR}/conf.py

${PYFORMEXDIR}/__init__.py: RELEASE
	sed -i 's|${VERSIONSTRING}|${NEWVERSIONSTRING}|' $@
	sed -i "/^Copyright/s|2004-....|2004-$$(date +%Y)|" $@

${LIBDIR}/configure.ac: RELEASE
	sed -i 's|^AC_INIT.*|AC_INIT(pyformex-lib,${RELEASE})|'  $@

${SPHINXDIR}/conf.py: RELEASE
	sed -i "s|^version =.*|version = '${VERSION}'|;s|^release =.*|release = '${RELEASE}'|" $@

setup.py: RELEASE
	sed -i "s|version='.*'|version='${RELEASE}'|" $@

# Stamp files with the version/release date

stamp: Stamp.template RELEASE
	${STAMP} -tStamp.template version=${VERSION} -sStamp.stamp

stampall: stamp
	${STAMP} -tStamp.stamp -i ${STAMPABLE}
	chmod +x ${EXECUTABLE}

printstampable:
	@for f in ${STAMPABLE}; do echo $$f; done

# Create the distribution
dist: ${LATEST}

${LATEST}: ${PKGDIR}/${PKGVER}
	ln -sfn ${PKGVER} ${PKGDIR}/${LATEST}

${PKGDIR}/${PKGVER}: revision version MANIFEST.in
	@echo "Creating ${PKGDIR}/${PKGVER}"
	python setup.py sdist --no-defaults | tee makedist.log

MANIFEST.in: MANIFEST.py
	python $< >$@

# Publish the distribution to our ftp server and berlios
publocal: 
	rsync -ltv ${PKGDIR}/${PKGVER} ${PKGDIR}/${LATEST} ${FTPLOCAL}

pub:
	rsync -ltv ${PKGDIR}/${PKGVER} ${PKGDIR}/${LATEST} ${FTPPYFORMEX}

# Register with the python package index
register:
	python setup.py register

upload:
	python setup.py sdist upload --show-response

# Tag the release in the svn repository
tag:
	svn copy svn+ssh://svn.berlios.de/svnroot/repos/pyformex/trunk svn+ssh://svn.berlios.de/svnroot/repos/pyformex/tags/release-${RELEASE} -m "Tagging the ${RELEASE} release of the 'pyFormex' project."

# Creates statistics
stats:
	./create_revision_graph
	./sloc.py

# Create the Sphinx documentation
html:
	make -C ${SPHINXDIR} html
	@echo "Remember to do 'make svndoc' to make the new docs available in pyformex-svn"

svndoc:
	make -C ${SPHINXDIR} svndoc

pdf:
	make -C ${SPHINXDIR} pdf

pubdoc:
	make -C ${SPHINXDIR} pub

pubpdf:
	make -C ${SPHINXDIR} pubpdf

# End
