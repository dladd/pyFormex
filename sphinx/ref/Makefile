# $Id$
##
##  This file is part of pyFormex 0.8.8  (Sun Nov  4 15:24:17 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
# Makefile for pyFormex reference manual sections
#
# The pyFormex project uses Sphinx for building its documentation.
# This Makefile creates an autodoc Sphinx file from the pyFormex source.
# It works using the py2rst.py script, because the autodoc functions
# of Sphinx are not yet powerful enough.
# 

COREMODULES= adjacency arraytools collection config connectivity coords \
  elements fileread filewrite \
  flatkeydb formex geometry geomtools mesh mydict odict olist project \
  script sendmail simple timer utils
GUIMODULES= actors camera canvas colors colorscale decors draw gluttext \
  image imageViewer imagearray marks menu\
  scriptMenu toolbar viewport widgets 
PLUGINMODULES= calpy_itf cameratools curve datareader dxf export \
  fe fe_abq fe_post flavia \
  inertia isopar lima mesh_ext nurbs objects plot2d polygon \
  postproc properties pyformex_gts section2d sectionize \
  tetgen tools trisurface turtle units 
#AUTOREF= $(COREMODULES) $(addprefix gui., $(GUIMODULES)) $(addprefix plugins., $(PLUGINMODULES))
AUTOREF= $(COREMODULES) $(GUIMODULES) $(PLUGINMODULES)
AUTOSRC= $(addsuffix .rst, ${AUTOREF})

MANUALREF = #geometry
MANUALSRC = $(addsuffix .rst, ${MANUALREF})

PY2RST=../py2rst.py

.PHONY: ref

# where to search for source files
PYFORMEXDIR=../pyformex
vpath %.py $(addprefix ../../pyformex, / /gui/ /plugins/)


all: ${AUTOSRC}

clean:
	rm -f ${AUTOSRC}

%.rst: %.py
	${PY2RST} $< > $@


# End
