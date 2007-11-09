#!/usr/bin/env python
# $Id$

"""selection.py

Operate on selected objects from the globals.
"""

import globaldata as GD
from gui import actors
from gui.draw import *
from plugins.surface import *

import commands, os, timer


class Selection(list):

    def __init__(self,data):

        list.__init__(data)
# The selection should only be set by setSelection()!!
selection = []  # a list of names of the currently selected Formices


##################### select, read and write ##########################

def setSelection(namelist):
    """Set the selection to a list of names.

    You should make sure this is a list of Formex names!
    This will also set oldvalues to the corresponding Formices.
    """
    global selection
    if type(namelist) == str:
        namelist = [ namelist ]
    selection = namelist


def clearSelection():
    """Clear the current selection."""
    setSelection([])


def changeSelection(newvalues):
    """Replace the current values of selection by new ones."""
    export(dict(zip(selection,newvalues)))


def checkSelection(single=False,warn=True):
    """Check that we have a current selection.

    Returns the list of Surfaces corresponding to the current selection.
    If single==True, the selection should hold exactly one Surface name and
    a single Surface instance is returned.
    If there is no selection, or more than one in case of single==True,
    an error message is displayed and an empty list is returned.
    """
    if not selection:
        if warn:
            warning("No Surface selected")
        return []
    if single and len(selection) > 1:
        if warn:
            warning("You should select exactly one Surface")
        return []
    if single:
        return named(selection[0])
    else:
        return map(named,selection)
    

def askSelection(mode=None):
    """Show the names of known formices and let the user select (one or more).

    This just returns a list of selected Formex names.
    It does not set the current selection. (see makeSelection)
    """
    return widgets.Selection(listAll(clas=surface.Surface),'Known Surfaces',
                             mode,sort=True,selected=selection).getResult()


def makeSelection(mode='multi'):
    """Interactively sets the current selection."""
    setSelection(askSelection(mode))
    drawSelection()


def drawSelection(*args,**kargs):
    """Draws the current selection.

    Any arguments are passed to draw()"""
    clear()
    if selection:
        for n in selection:
            print named(n)
        draw(selection,*args,**kargs)
        #if show_numbers:
        #    showSelectionNumbers()


def forgetSelection():
    if selection:
        forget(selection)


def read_Surface(fn):
    GD.message("Reading file %s" % fn)
    t = timer.Timer()
    S = Surface.read(fn)
    GD.message("Read surface with %d vertices, %d edges, %d triangles in %s seconds" % (S.ncoords(),S.nedges(),S.nelems(),t.seconds()))
    return S


def readSelection(select=True,draw=True,multi=True):
    """Read a Surface (or list) from asked file name(s).

    If select is True (default), this becomes the current selection.
    If select and draw are True (default), the selection is drawn.
    """
    types = [ 'Surface Files (*.gts *.stl *.off *.neu *.smesh)', 'All Files (*)' ]
    fn = askFilename(GD.cfg['workdir'],types,exist=True,multi=multi)
    if not multi:
        fn = [ fn ]
    if fn:
        chdir(fn[0])
        names = map(utils.projectName,fn)
        GD.gui.setBusy()
        F = map(read_Surface,fn)
        GD.gui.setBusy(False)
        export(dict(zip(names,F)))
        if select:
            GD.message("Set selection to %s" % str(names))
            for n in names:
                print "%s = %s" % (n,named(n))
            setSelection(names)
            if draw:
                drawSelection()
    return fn

if __name__ == "__main__":
    print __doc__

# End
