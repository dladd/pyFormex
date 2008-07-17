#  -*- Makefile -*-  for creating pyFormex releases
# $Id$
##
## This file is part of pyFormex 0.7 Release Fri Apr  4 18:41:11 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
#

include RELEASE

PKGNAME= pyformex

PYFORMEXDIR= pyformex

MANDIR= ${PYFORMEXDIR}/manual
LIBDIR= ${PYFORMEXDIR}/lib
DOCDIR= ${PYFORMEXDIR}/doc

SOURCE= \
	$(wildcard ${PYFORMEXDIR}/*.py) \
	$(wildcard ${PYFORMEXDIR}/gui/*.py) \
	$(wildcard ${PYFORMEXDIR}/plugins/*.py) \

EXAMPLES= \
	$(wildcard ${PYFORMEXDIR}/examples/*.py) \
	$(wildcard ${PYFORMEXDIR}/examples/Analysis/*.py) \
	$(wildcard ${PYFORMEXDIR}/examples/Demos/*.py) \

EXAMPLEDATA= $(wildcard ${PYFORMEXDIR}/examples/*.db)


OTHERSTAMPABLE= setup.py \
	${PYFORMEXDIR}/pyformexrc \
	${EXAMPLEDATA} \
	${LIBDIR}/Makefile \
	${addprefix ${DOCDIR}/, README ReleaseNotes}

NONSTAMPABLE= ${DOC}/COPYING 

STAMPABLE= ${SOURCE} ${EXAMPLES} ${OTHERSTAMPABLE}


STAMP= stamp 
VERSIONSTRING= _version_ = .*
NEWVERSIONSTRING= _version_ = "${RELEASE}"

PKGVER= ${PKGNAME}-${RELEASE}.tar.gz
PKGDIR= dist
LATEST= pyformex-latest.tar.gz

# outr local ftp server
FTPLOCAL=bumps:/home/ftp/pub/pyformex
# ftp server on pyformex website
FTPPYFORMEX=bverheg@shell.berlios.de:/home/groups/ftp/pub/pyformex

.PHONY: dist pub distclean pydoc manual minutes website stamp dist.stamped version tag

############ Creating Distribution ##################

default:
	@echo Please specify a target

distclean:
	alldirs . "rm -f *~"

# Create the manual
manual:
	make -C ${MANDIR}

# Create the C library
lib:
	make -C ${LIBDIR}

# Create the pydoc html files
pydoc:
	make -C ${DOCDIR}

# Create the minutes of the user meeting
minutes: 
	make -C user

# Create the website
website: 
	make -C website


# Set a new version

version: ${PYFORMEXDIR}/__init__.py ${MANDIR}/pyformex.tex setup.py ${LIBDIR}/configure.ac

${PYFORMEXDIR}/__init__.py: RELEASE
	sed -i 's|${VERSIONSTRING}|${NEWVERSIONSTRING}|' $@

${MANDIR}/pyformex.tex: RELEASE
	sed -i 's|\\release{.*}|\\release{${RELEASE}}|;s|\\setshortversion{.*}|\\setshortversion{${VERSION}}|;'  $@

${LIBDIR}/configure.ac: RELEASE
	sed -i 's|^AC_INIT.*|AC_INIT(pyformex-lib,${RELEASE})|'  $@

setup.py: RELEASE
	sed -i "s|version='.*'|version='${RELEASE}'|" $@

# Stamp files with the version/release date

stamp: Stamp.template RELEASE
	${STAMP} -tStamp.template version=${VERSION} -sStamp.stamp

stampall: stamp
	${STAMP} -tStamp.stamp -i ${STAMPABLE}

printstampable:
	@for f in ${STAMPABLE}; do echo $$f; done

# Create the distribution
dist: ${LATEST}

${LATEST}: ${PKGDIR}/${PKGVER}
	ln -sfn ${PKGVER} ${PKGDIR}/${LATEST}

${PKGDIR}/${PKGVER}: version MANIFEST.in
	@echo "Creating ${PKGDIR}/${PKGVER}"
	rm -f MANIFEST
	python setup.py sdist

# Publish the distribution to our ftp server and berlios
publocal: 
	rsync -lt ${PKGDIR}/${PKGVER} ${PKGDIR}/${LATEST} ${FTPLOCAL}

pub:
	rsync -lt ${PKGDIR}/${PKGVER} ${PKGDIR}/${LATEST} ${FTPPYFORMEX}

# Tag the release in the svn repository
tag:
	svn copy svn+ssh://svn.berlios.de/svnroot/repos/pyformex/trunk svn+ssh://svn.berlios.de/svnroot/repos/pyformex/tags/release-${RELEASE} -m "Tagging the ${RELEASE} release of the 'pyFormex' project."

# End
