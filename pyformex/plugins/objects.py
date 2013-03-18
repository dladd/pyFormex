# $Id$
##
##  This file is part of pyFormex 0.8.9  (Fri Nov  9 10:49:51 CET 2012)
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

"""Selection of objects from the global dictionary.

This is a support module for other pyFormex plugins.
"""
from __future__ import print_function

import pyformex as pf

from coords import bbox
from script import named
from gui.draw import drawBbox,_I
import geomfile
import odict

from copy import deepcopy


class Objects(object):
    """A selection of objects from the pyFormex Globals().

    The class provides facilities to filter the global objects by their type
    and select one or more objects by their name(s). The values of these
    objects can be changed and the changes can be undone.
    """

    def __init__(self,clas=None,like=None,filter=None,namelist=[]):
        """Create a new selection of objects.

        If a filter is given, only objects passing it will be accepted.
        The filter will be applied dynamically on the dict.

        If a list of names is given, the current selection will be set to
        those names (provided they are in the dictionary).
        """
        self.clas = clas
        self.like = like
        self.filter = filter
        self.names = []
        self.values = []
        self.clear()
        if namelist:
            self.set(namelist)


    def object_type(self):
        """Return the type of objects in this selection."""
        if self.clas:
            return self.clas.__name__+' '
        else:
            return ''


    def set(self,names):
        """Set the selection to a list of names.

        namelist can be a single object name or a list of names.
        This will also store the current values of the variables.
        """
        if type(names) == str:
            names = [ names ]
        self.names = [ s for s in names if type(s) == str ]
        self.values = map(named,self.names)


    def append(self,name,value=None):
        """Add a name,value to a selection.

        If no value is given, its current value is used.
        If a value is given, it is exported.
        """
        self.names.append(name)
        if value is None:
            value = named(name)
        else:
            export({name:value})
        self.values.append(value)


    def clear(self):
        """Clear the selection."""
        self.set([])


    def __getitem__(self,i):
        """Return selection item i"""
        return self.names[i]


    def listAll(self):
        """Return a list with all selectable objects.

        This lists all the global names in pyformex.PF that match
        the class and/or filter (if specified).
        """
        return listAll(clas=self.clas,like=self.like,filtr=self.filter)


    def selectAll(self):
        self.set(self.listAll())


    def remember(self,copy=False):
        """Remember the current values of the variables in selection.

        If copy==True, the values are copied, so that the variables' current
        values can be changed inplace without affecting the remembered values.
        """
        self.values = map(named,self.names)
        if copy:
            self.values = map(deepcopy,self.values)


    def changeValues(self,newvalues):
        """Replace the current values of selection by new ones.

        The old values are stored locally, to enable undo operations.

        This is only needed to change the values of objects that can not
        be changed inplace!
        """
        self.remember()
        export2(self.names,newvalues)


    def undoChanges(self):
        """Undo the last changes of the values."""
        export2(self.names,self.values)


    def check(self,single=False,warn=True):
        """Check that we have a current selection.

        Returns the list of Objects corresponding to the current selection.
        If single==True, the selection should hold exactly one Object name and
        a single Object instance is returned.
        If there is no selection, or more than one in case of single==True,
        an error message is displayed and None is returned
        """
        self.names = [ n for n in self.names if n in pf.PF ]
        if len(self.names) == 0:
            if warn:
                warning("No %sobjects were selected" % self.object_type())
            return None
        if single and len(self.names) > 1:
            if warn:
                warning("You should select exactly one %sobject" %  self.object_type())
            return None
        if single:
            return named(self.names[0])
        else:
            return map(named,self.names)


    def odict(self):
        """Return the currently selected items as a dictionary.

        Returns an ODict with the currently selected objects in the order
        of the selection.names.
        """
        return odict.ODict(zip(self.names,self.check(warn=False)))


    def ask(self,mode='multi'):
        """Show the names of known objects and let the user select one or more.

        mode can be set to'single' to select a single item.
        Return a list with the selected names, possibly empty (if nothing
        was selected by the user), or None if there is nothing to choose from.
        This also sets the current selection to the selected names, unless
        the return value is None, in which case the selection remains unchanged.
        """
        choices = self.listAll()
        if not choices:
            return None
        res = widgets.ListSelection(
            caption='Known %sobjects' % self.object_type(),
            choices=self.listAll(),
            default=self.names,
            sort=True).getResult()
        if res is None:
            res = []
        self.set(res)
        return res


    def ask1(self):
        """Select a single object from the list.

        Returns the object, not its name!
        """
        if self.ask('single'):
            return named(self.names[0])
        else:
            return None


    def forget(self):
        """Remove the selection from the globals."""
        forget(self.names)
        self.clear()


    def keep(self):
        """Remove everything except the selection from the globals."""
        forget([ n for n in self.listAll() if not n in self.names ])


    def printval(self):
        """Print the selection."""
        objects = self.check()
        if objects:
            for n,o in zip(self.names,objects):
                print("%s = %s" % (n,str(o)))


    def printbbox(self):
        """Print the bbox of the current selection."""
        objects = self.check()
        if objects:
            for n,o in zip(self.names,objects):
                bb = o.bbox()
                pf.message("* %s (%s): bbox [%s, %s]" % (n,o.__class__.__name__,bb[0],bb[1]))
            if len(self.names) > 1:
                pf.message("** Overal bbox: [%s, %s]" % (bb[0],bb[1]))


    def writeToFile(self,filename):
        """Write objects to a geometry file."""
        objects = self.odict()
        if objects:
            writeGeomFile(filename,objects)


    def readFromFile(self,filename):
        """Read objects from a geometry file."""
        res = readGeomFile(filename)
        export(res)
        self.set(res.keys())


###################### Drawable Objects #############################

from gui.draw import *

# Default Annotations

def draw_object_name(n):
    """Draw the name of an object at its center."""
    return drawText3D(named(n).center(),n)

def draw_elem_numbers(n):
    """Draw the numbers of an object's elements."""
    return drawNumbers(named(n),color='blue')

def draw_nodes(n):
    """Draw the nodes of an object."""
    return draw(named(n).coords,nolight=True,wait=False)

def draw_node_numbers(n):
    """Draw the numbers of an object's nodes."""
    return drawNumbers(named(n).coords,color='red')

def draw_free_edges(n):
    """Draw the feature edges of an object."""
    return drawFreeEdges(named(n),color='black')

def draw_bbox(n):
    """Draw the bbox of an object."""
    return drawBbox(named(n))


class DrawableObjects(Objects):
    """A selection of drawable objects from the globals().

    This is a subclass of Objects. The constructor has the same arguments
    as the Objects class, plus the following:

    - `annotations`: a set of functions that draw annotations of the objects.
      Each function should take an object name as argument, and draw the
      requested annotation for the named object. If the object does not have
      the annotation, it should be silently ignored.
      Default annotation functions available are:

      - draw_object_name
      - draw_elem_numbers
      - draw_nodes
      - draw_node_numbers
      - draw_bbox

      No annotation functions are activated by default.

    """
    def __init__(self,**kargs):
        Objects.__init__(self,**kargs)
        self.autodraw = False
        self.shrink = None
        self.annotations = set() # Active annotations
        self._annotations = {} # Drawn annotations
        self._actors = []


    def ask(self,mode='multi'):
        """Interactively sets the current selection."""
        new = Objects.ask(self,mode)
        if new is not None:
            self.draw()
        return new


    def draw(self,**kargs):
        clear()
        pf.debug("Drawing SELECTION: %s" % self.names,pf.DEBUG.DRAW)
        self._actors = draw(self.names,clear=False,shrink=self.shrink,wait=False,**kargs)
        for f in self.annotations:
            pf.debug("Drawing ANNOTATION: %s" % f,pf.DEBUG.DRAW)
            self.drawAnnotation(f)


    def drawChanges(self):
        """Draws old and new version of a Formex with different colors.

        old and new can be a either Formex instances or names or lists thereof.
        old are drawn in yellow, new in the current color.
        """
        self.draw()
        draw(self.values,color='yellow',bbox=None,clear=False,shrink=self.shrink,wait=False)


    def undoChanges(self):
        """Undo the last changes of the values."""
        Objects.undoChanges(self)
        self.draw()


    def toggleAnnotation(self,f,onoff=None):
        """Toggle the display of an annotation On or Off.

        If given, onoff is True or False.
        If no onoff is given, this works as a toggle.
        """
        if onoff is None:
            # toggle
            active = f not in self.annotations
        else:
            active = onoff
        if active:
            self.annotations.add(f)
            self.drawAnnotation(f)
        else:
            self.annotations.discard(f)
            self.removeAnnotation(f)


    def drawAnnotation(self,f):
        """Draw some annotation for the current selection."""
        self._annotations[f] = [ f(n) for n in self.names ]


    def removeAnnotation(self,f):
        """Remove the annotation f."""
        if f in self._annotations:
            # pf.canvas.removeAnnotation(self._annotations[f])
            # Use removeAny, because some annotations are not canvas
            # annotations but actors!
            pf.canvas.removeAny(self._annotations[f])
            pf.canvas.update()
            del self._annotations[f]


    def editAnnotations(self,ontop=None):
        """Edit the annotation properties

        Currently only changes the ontop attribute for all drawn
        annotations. Values: True, False or '' (toggle).
        Other values have no effect.

        """
        for f in self._annotations.values():
            if ontop in [ True, False, '' ]:
                if not isinstance(f,list):
                   f = [f]
                for a in f:
                    if ontop == '':
                        ontop = not a.ontop
                    print(a,ontop)
                    a.ontop = ontop


    def hasAnnotation(self,f):
        """Return the status of annotation f"""
        return f in self.annotations
    def hasNames(self):
        return self.hasAnnotation(draw_object_name)
    def hasNumbers(self):
        return self.hasAnnotation(draw_elem_numbers)
    def hasNodeNumbers(self):
        return self.hasAnnotation(draw_node_numbers)
    def hasFreeEdges(self):
        return self.hasAnnotation(draw_free_edges)
    def hasNodeMarks(self):
        return self.hasAnnotation(draw_nodes)
    def hasBbox(self):
        return self.hasAnnotation(draw_bbox)

    def toggleNames(self,onoff=None):
        self.toggleAnnotation(draw_object_name,onoff)
    def toggleNumbers(self,onoff=None):
        self.toggleAnnotation(draw_elem_numbers,onoff)
    def toggleNodeNumbers(self,onoff=None):
        self.toggleAnnotation(draw_node_numbers,onoff)
    def toggleFreeEdges(self,onoff=None):
        self.toggleAnnotation(draw_free_edges,onoff)
    def toggleNodes(self,onoff=None):
        self.toggleAnnotation(draw_nodes,onoff)
    def toggleBbox(self,onoff=None):
        self.toggleAnnotation(draw_bbox,onoff)


    def setProp(self,prop=None):
        """Set the property of the current selection.

        prop should be a single integer value or None.
        If None is given, a value will be asked from the user.
        If a negative value is given, the property is removed.
        If a selected object does not have a setProp method, it is ignored.
        """
        objects = self.check()
        if objects:
            if prop is None:
                res = askItems([['property',0]],
                               caption = 'Set Property Number for Selection (negative value to remove)')
                if res:
                    prop = int(res['property'])
                    if prop < 0:
                        prop = None
            for o in objects:
                if hasattr(o,'setProp'):
                    o.setProp(prop)
            self.draw()


    def delProp(self):
        """Delete the property of the current selection.

        This well reset the `prop` attribute of all selected objects
        to None.
        """
        objects = self.check()
        if objects:
            for o in objects:
                if hasattr(o,'prop'):
                    o.prop=None
            self.draw()




if __name__ == "draw":
    # If executed as a pyformex script
    pf.debug('Reloading module %s' % __file__)

elif __name__ == "__main__":
    print(__doc__)

# End
