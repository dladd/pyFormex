# Makefile for the pyformex user meeting
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

SRC= $(wildcard pyformex-user-meeting-*.rst)
ALLPDF= ${SRC:.rst=.pdf}
PDF= $(wildcard pyformex-user-meeting-*.pdf)
RSYNC= rsync -av -e ssh
ACCOUNT= bverheg@shell.berlios.de

default:
	@echo "You should use an explicit target, e.g. make #"
	@echo "where # is the number of the meeting."

all: ${ALLPDF}


%: pyformex-user-meeting-%.pdf
	@echo "Done"

%.pdf: %.rst
	rst2pdf $<

# Transfers the files to BerliOs
pub: 
	${RSYNC} ${PDF} ${ACCOUNT}:/home/groups/ftp/pub/pyformex


clean:

mrproper: clean
	rm -rf *.pdf