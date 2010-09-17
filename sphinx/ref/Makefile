# $Id$
#
# Makefile for pyFormex reference manual sections
#
# The pyFormex project uses Sphinx for building its documentation.
# This Makefile creates an autodoc Sphinx file from the pyFormex source.
# It works using the py2rst.py script, because the autodoc functions
# of Sphinx are not yet powerful enough.
# 

COREMODULES= arraytools collection config connectivity coords elements  \
  flatkeydb formex geometry misc mydict odict olist project \
  script sendmail simple timer utils
GUIMODULES= actors camera canvas colors colorscale decors draw gluttext \
  image imageViewer imagearray imagecolor marks menu\
  scriptMenu toolbar viewport widgets 
PLUGINMODULES= curve datareader dxf export fe fe_abq fe_post geomtools \
  inertia isopar lima mesh nurbs objects postproc properties section2d \
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

#gui.%.rst: %.py ${PY2RST}
#	${PY2RST} $< > $@

#plugins.%.rst: %.py ${PY2RST}
#	${PY2RST} $< > $@

%.rst: %.py ${PY2RST}
	${PY2RST} $< > $@


# End
