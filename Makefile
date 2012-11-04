# $Id$
##
##  This file is part of pyFormex 0.8.8  (Sun Nov  4 17:22:49 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2012 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
# Makefile for creating pyFormex releases
#

include RELEASE

PKGNAME= pyformex

PYFORMEXDIR= pyformex

LIBDIR= ${PYFORMEXDIR}/lib
DOCDIR= ${PYFORMEXDIR}/doc
BINDIR= ${PYFORMEXDIR}/bin
EXTDIR= ${PYFORMEXDIR}/extra
SPHINXDIR= sphinx

SOURCE= ${PYFORMEXDIR}/pyformex \
	$(wildcard ${PYFORMEXDIR}/*.py) \
	$(wildcard ${PYFORMEXDIR}/gui/*.py) \
	$(wildcard ${PYFORMEXDIR}/plugins/*.py) \
	$(wildcard ${LIBDIR}/*.py) \

LIBSOURCE= ${addprefix ${LIBDIR}/, drawgl_.c misc_.c nurbs_.c} 
LIBOBJECTS= $(CSOURCE:.c=.o)
LIBOBJECTS= $(CSOURCE:.c=.so)

BINSOURCE= \
	$(wildcard ${BINDIR}/*.awk) \
	${addprefix ${BINDIR}/, gambit-neu gambit-neu-hex} \

EXTSOURCE= \
	$(wildcard ${EXTDIR}/*/README*) \
	$(wildcard ${EXTDIR}/*/Makefile) \
	$(wildcard ${EXTDIR}/*/*.rst) \
	$(wildcard ${EXTDIR}/*/install.sh) \
	$(wildcard ${EXTDIR}/*/*.h) \
	$(wildcard ${EXTDIR}/*/*.c) \
	$(wildcard ${EXTDIR}/*/*.cc) \
	$(wildcard ${EXTDIR}/*/*.py) \
	${addprefix ${EXTDIR}/pygl2ps/, gl2ps.i setup.py} \

EXAMPLES= \
	$(wildcard ${PYFORMEXDIR}/examples/*.py) \
	$(wildcard ${PYFORMEXDIR}/examples/Demos/*.py) \

EXAMPLEDATA= $(wildcard ${PYFORMEXDIR}/data/*.db)

DOCSOURCE= \
	$(wildcard ${SPHINXDIR}/*.rst) \
	$(wildcard ${SPHINXDIR}/*.py) \
	$(wildcard ${SPHINXDIR}/*.inc) \
	$(wildcard ${SPHINXDIR}/static/scripts/[A-Z]*.py) \
	${SPHINXDIR}/Makefile \
	${SPHINXDIR}/ref/Makefile

EXECUTABLE= ${PYFORMEXDIR}/pyformex ${PYFORMEXDIR}/sendmail.py \
	${BINDIR}/read_abq_inp.awk \
	pyformex-viewer \
	${SPHINXDIR}/py2rst.py


OTHERSTAMPABLE= README Makefile ReleaseNotes \
	manifest.py setup.py \
	${PYFORMEXDIR}/pyformexrc \
	${EXAMPLEDATA} \
	$(wildcard ${DOCDIR}/*.rst)

NONSTAMPABLE= COPYING 

STAMPABLE= $(filter-out ${PYFORMEXDIR}/template.py,${SOURCE}) \
	${EXECUTABLE} ${CSOURCE} ${EXAMPLES} ${DOCSOURCE} ${BINSOURCE} \
	$(filter-out ${EXTDIR}/pygl2ps/gl2ps_wrap.c,${EXTSOURCE}) \
	${OTHERSTAMPABLE}

STATICSTAMPABLE= Description History HOWTO-dev.rst MANIFEST.py add_Id \
	create_revision_graph install-pyformex-svn-desktop-link \
	pyformex-viewer searchpy sloc.py slocstats.awk \
	user/Makefile $(wildcard user/*.rst) \
	website/Makefile $(wildcard website/scripts/*.py) \
	$(wildcard website/src/examples/*.txt)

STATICDIRS= pyformex/data/README pyformex/icons/README \
	pyformex/lib/README \
	screenshots/README \
	sphinx/images/README sphinx/static/scripts/README \
	website/README website/images/README website/src/README \
	website/src/examples/README

STAMP= stamp 
VERSIONSTRING= __version__ = .*
NEWVERSIONSTRING= __version__ = "${RELEASE}"

PKGVER= ${PKGNAME}-${RELEASE}.tar.gz
PKGDIR= dist
PUBDIR= ${PKGDIR}/pyformex
LATEST= ${PKGNAME}-latest.tar.gz

# our local ftp server
FTPLOCAL= bumps:/var/ftp/pub/pyformex
# ftp server on Savannah
FTPPUB= bverheg@dl.sv.nongnu.org:/releases/pyformex/

.PHONY: dist pub clean html latexpdf pubdoc minutes website dist.stamped version tag register bumprelease bumpversion stampall stampstatic stampstaticdirs

##############################

default:
	@echo Please specify a target

clean:
	alldirs . "rm -f *~" 

distclean: clean
	alldirs . "rm -f *.pyc *.so"

# Create the C library
lib: 
	python setup.py build_ext
	find build -name '*.so' -exec mv {} pyformex/lib \;
	rm -rf build

# Create the C library with debug option
#libdebug: ${LIBDIR}/Makefile
#	make -C ${LIBDIR} debug

# Create the C library without debug option
#libnodebug: ${LIBDIR}/Makefile
#	make -C ${LIBDIR} nodebug

# Clean C library
#libreset: ${LIBDIR}/Makefile
#	make -C ${LIBDIR} reset

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

version: ${PYFORMEXDIR}/__init__.py setup.py ${SPHINXDIR}/conf.py

${PYFORMEXDIR}/__init__.py: RELEASE
	sed -i 's|${VERSIONSTRING}|${NEWVERSIONSTRING}|' $@
	sed -i "/^Copyright/s|2004-....|2004-$$(date +%Y)|" $@

${SPHINXDIR}/conf.py: RELEASE
	sed -i "s|^version =.*|version = '${VERSION}'|;s|^release =.*|release = '${RELEASE}'|" $@

setup.py: RELEASE
	sed -i "s|version='.*'|version='${RELEASE}'|" $@

# Stamp files with the version/release date

Stamp.stamp: Stamp.template RELEASE
	${STAMP} -t$< header="This file is part of pyFormex ${VERSION}  ($$(env LANG=C date))" -s$@

stampall: Stamp.stamp
	${STAMP} -t$< -i ${STAMPABLE}
	chmod +x ${EXECUTABLE}
# this should be fixed in stamp !


printstampable:
	@for f in ${STAMPABLE}; do echo $$f; done

Stamp.static: Stamp.template
	${STAMP} -t$< header='This file is part of the pyFormex project.' -s$@

stampstatic: Stamp.static
	${STAMP} -t$< -i ${STATICSTAMPABLE}

Stamp.staticdir: Stamp.template
	${STAMP} -t$< header='The files in this directory are part of the pyFormex project.' -s$@

stampstaticdirs: Stamp.staticdir
	${STAMP} -t$< -i ${STATICDIRS}

# Create the distribution
dist: manpages ${PKGDIR}/${LATEST}

${PKGDIR}/${LATEST}: ${PKGDIR}/${PKGVER}
	ln -sfn ${PKGVER} ${PKGDIR}/${LATEST}

${PKGDIR}/${PKGVER}: revision version
	@echo "Creating ${PKGDIR}/${PKGVER}"
	python setup.py sdist --no-defaults | tee makedist.log
	python manifest_check.py

# Create all our manpages
manpages:
	make -C pyformex/doc manpages
	make -C pyformex/extra manpages


# Publish the distribution to our ftp server

publocal: ${PKGDIR}/${LATEST}
	rsync -ltv ${PKGDIR}/${PKGVER} ${PKGDIR}/${LATEST} ${FTPLOCAL}

sign: ${PUBDIR}/${PKGVER}.sig

${PUBDIR}/${PKGVER}.sig:
	mv ${PKGDIR}/${PKGVER} ${PKGDIR}/${LATEST} ${PUBDIR}
	cd ${PUBDIR}; gpg -b --use-agent ${PKGVER}

pubn: ${PUBDIR}/${PKGVER}.sig
	rsync ${PUBDIR}/* ${FTPPUB} -rtlvn 

pub: ${PUBDIR}/${PKGVER}.sig
	rsync ${PUBDIR}/* ${FTPPUB} -rtlv

# Register with the python package index
register:
	python setup.py register

upload:
	python setup.py sdist upload --show-response

# Tag the release in the svn repository
# THIS WILL ONLY WORK IF YOU HAVE YOUR USER NAME CONFIGURED IN .ssh/config
tag:
	svn copy svn+ssh://svn.savannah.nongnu.org/pyformex/trunk svn+ssh://svn.savannah.nongnu.org/pyformex/tags/release-${RELEASE} -m "Tagging the ${RELEASE} release of the 'pyFormex' project."

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

latexpdf:
	make -C ${SPHINXDIR} latexpdf

# Publish the documentation on the website

pubdoc:
	make -C ${SPHINXDIR} pubdoc


listwww:
	cd www; cvs ls | grep '^?'; cd ..
	@echo "Add the ? files by hand!"

commit:
	cd www; cvs commit; cd ..

# Make the PDF manual available for download

pubpdf:
	make -C ${SPHINXDIR} pubpdf


# Test all modules
# Currently this tests only the core modules
testall:
	cd pyformex; for f in *.py; do pyformex --testmodule $${f%.py}; done

# End
