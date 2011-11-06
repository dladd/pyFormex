# Makefile for the pyformex WEB site
##
##  This file is part of the pyFormex project.
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  https://savannah.nongnu.org/projects/pyformex/
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

HOMEPARTS= home_header.inc home_footer.inc
HTDOCS= install.html links.html screenshots.html support.html pyformexinuse.html pyformex.html pyformex.css favicon.ico
EXAMPLES= examples/
IMAGES= images/
SCRIPTS= scripts/
RSYNC= rsync -av -e ssh --delete --exclude .svn

SERVER= shell.berlios.de
TARGETDIR= ${SERVER}:/home/groups/pyformex/htdocs

.PHONY: html pub all

html: 
	r2w
	./split_home
	./update-news noupdate


all:	pub


# Transfers the files to BerliOs
pubtest: $(HOMEPARTS)
	$(RSYNC) --exclude src . ${TARGETDIR} -n

pubimages: 
	$(RSYNC) $(IMAGES) ${TARGETDIR}/images

pub: 
	$(RSYNC) $(HTDOCS) ${TARGETDIR}
	$(RSYNC) $(EXAMPLES) ${TARGETDIR}/examples
	$(RSYNC) $(IMAGES) ${TARGETDIR}/images
	$(RSYNC) $(SCRIPTS) ${TARGETDIR}/scripts
