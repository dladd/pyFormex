# $Id$
##
##  This file is part of pyFormex 0.8.5  (Sun Dec  4 21:24:46 CET 2011)
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

COREMODULES= arraytools collection config connectivity coords elements  \
  flatkeydb formex geometry mesh mydict odict olist project \
  script sendmail simple timer utils
GUIMODULES= actors camera canvas colors colorscale decors draw gluttext \
  image imageViewer imagearray imagecolor marks menu\
  scriptMenu toolbar viewport widgets 
PLUGINMODULES= curve datareader dxf export fe fe_abq fe_post flavia geomtools \
  inertia isopar lima mesh_ext nurbs objects plot2d postproc properties \
  section2d tetgen tools trisurface turtle units
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

#gui.%.rst: %.py ${PY2RST}
#	${PY2RST} $< > $@

#plugins.%.rst: %.py ${PY2RST}
#	${PY2RST} $< > $@

%.rst: %.py
	${PY2RST} $< > $@


# End
