# $Id$
##
##  This file is part of pyFormex
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
# Makefile for pyFormex documentation
#
# The pyFormex project uses Sphinx for building its documentation.
# This Makefile was create with sphinx-quickstart and then hand-edited.
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         =

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d _build/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .

MANDOCS= _build/html/*
RSYNC= rsync -v -e ssh
PDF= _build/latex/pyformex.pdf

SVNDOC= ../pyformex/doc/html
UPLOAD= shell.berlios.de:/home/groups
DOCUPLOAD= ${UPLOAD}/pyformex/htdocs/doc
FTPUPLOAD= ${UPLOAD}/ftp/pub/pyformex
FTPLOCAL= bumps.ugent.be:/var/ftp/pub/pyformex

.PHONY: help clean ref html dirhtml pickle json htmlhelp qthelp latex changes linkcheck doctest

# where to search for source files
PYFORMEXDIR=../pyformex
vpath %.py $(addprefix ../pyformex, / /gui/ /plugins/)

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  ref       to recreate the refman source files"
	@echo "  html      to make standalone HTML files"
	@echo "  dirhtml   to make HTML files named index.html in directories"
	@echo "  pickle    to make pickle files"
	@echo "  json      to make JSON files"
	@echo "  htmlhelp  to make HTML files and a HTML help project"
	@echo "  qthelp    to make HTML files and a qthelp project"
	@echo "  latex     to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  changes   to make an overview of all changed/added/deprecated items"
	@echo "  linkcheck to check all external links for integrity"
	@echo "  doctest   to run all doctests embedded in the documentation (if enabled)"
	@echo "  pdf       to make PDF manual"

clean:
	make -C ref clean
	rm -rf _build/*

html: prereq
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) _build/html
	rsync images/README  _build/html/_images
	@echo
	@echo "Build finished. The HTML pages are in _build/html."

dirhtml:
	$(SPHINXBUILD) -b dirhtml $(ALLSPHINXOPTS) _build/dirhtml
	@echo
	@echo "Build finished. The HTML pages are in _build/dirhtml."

pickle:
	$(SPHINXBUILD) -b pickle $(ALLSPHINXOPTS) _build/pickle
	@echo
	@echo "Build finished; now you can process the pickle files."

json:
	$(SPHINXBUILD) -b json $(ALLSPHINXOPTS) _build/json
	@echo
	@echo "Build finished; now you can process the JSON files."

htmlhelp:
	$(SPHINXBUILD) -b htmlhelp $(ALLSPHINXOPTS) _build/htmlhelp
	@echo
	@echo "Build finished; now you can run HTML Help Workshop with the" \
	      ".hhp project file in _build/htmlhelp."

qthelp:
	$(SPHINXBUILD) -b qthelp $(ALLSPHINXOPTS) _build/qthelp
	@echo
	@echo "Build finished; now you can run "qcollectiongenerator" with the" \
	      ".qhcp project file in _build/qthelp, like this:"
	@echo "# qcollectiongenerator _build/qthelp/pyFormex.qhcp"
	@echo "To view the help file:"
	@echo "# assistant -collectionFile _build/qthelp/pyFormex.qhc"

latex:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) _build/latex
	@echo
	@echo "Build finished; the LaTeX files are in _build/latex."
	@echo "Run \`make all-pdf' or \`make all-ps' in that directory to" \
	      "run these through (pdf)latex."

changes:
	$(SPHINXBUILD) -b changes $(ALLSPHINXOPTS) _build/changes
	@echo
	@echo "The overview file is in _build/changes."

linkcheck:
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) _build/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in _build/linkcheck/output.txt."

doctest:
	$(SPHINXBUILD) -b doctest $(ALLSPHINXOPTS) _build/doctest
	@echo "Testing of doctests in the sources finished, look at the " \
	      "results in _build/doctest/output.txt."


pdf: ${PDF}

${PDF}: #latex
	make -C _build/latex all-pdf


# Automatically created input files

prereq: ref pyformex.help

ref:
	make -C ref

pyformex.help: ../pyformex/pyformex
	../pyformex/pyformex --help > $@


# Transfer the html docs to doc
svndoc:
	${RSYNC} -a --exclude '.svn' --delete ${MANDOCS} ${SVNDOC} 

# Transfer the files to BerliOS
pub: 
	${RSYNC} -a --exclude '.svn' --delete ${MANDOCS} ${DOCUPLOAD}

pubpdf: ${PDF}
	${RSYNC} -t $< ${FTPUPLOAD}

# and to our ftp server
publocal: ${PDF}
	rsync -lt $< ${FTPLOCAL}

# End
