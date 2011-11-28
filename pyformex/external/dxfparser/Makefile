# $Id$
##
##  This file is part of pyFormex 0.8.5     Sun Nov  6 17:27:05 CET 2011
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

#
############# SET THESE TO SUIT YOUR INSTALLATION ####################

# !! An environment variable DESTDIR can be set to specify a path for
# !! the installation tree. All install paths specified hereafter will
# !! be relative to that installation path.

# root of the installation tree: this is a reasonable default
prefix:= /usr/local
# where to install the executable files
bindir:= $(prefix)/bin
# where to install library files
libdir:= $(prefix)/lib
# where to install the documentation
docdir:= $(prefix)/share/doc

#####################################################################

PKG:= dxfparser

BINDIR:= ${DESTDIR}${bindir}

INSTALL:= install
INSTALL_EXE:= ${INSTALL} -m 0755
INSTALL_DATA:= ${INSTALL} -m 0644

LFLAGS = -l dxflib

.PHONY: all build install clean


build: dxfparser

all: dxfparser
	gksu make install


dxfparser: dxfparser.o
	g++ $(LFLAGS) $< -o $@

dxfparser.o: dxfparser.h dxfparser.cc


install: dxfparser
	${INSTALL_EXE} $< ${BINDIR}

clean:
	rm -f dxfparser dxfparser.o

# End
