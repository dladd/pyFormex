#!/usr/bin/env python pyformex
# $Id$
##
##  This file is part of pyFormex 0.7.3 Release Tue Dec 30 20:45:35 2008
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Website: http://pyformex.berlios.de/
##  Copyright (C) Benedict Verhegghe (bverheg@users.berlios.de) 
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
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.
##

"""objects.py

This is a support module for other pyFormex plugins.
"""

import pyformex as GD

from coords import bbox
from formex import Formex
from plugins.surface import TriSurface
from copy import deepcopy


class Objects(object):
    """A selection of objects from the globals().

    The class provides facilities to filter the global objects by their type
    and select one or more objects by their name(s). The values of these
    objects can be changed and the changes can be undone.
    """

    def __init__(self,clas=None,filter=None,namelist=[]):
        """Create a new selection of objects.

        If a dict is given, objects will be selected from this dict, else
        from the global pyFormex dict GD.PF.
        If a filter is given, only objects passing it will be accepted.
        The filter will be applied dynamically on the dict.

        If a list of names is given, the current selection will be set to
        those names (provided they are in the dictionary).
        """
        self.clas = clas
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
    

    def selectAll(self):
        self.set(self.listAll(clas=self.clas))


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


    def ask(self,mode='multi'):
        """Show the names of known objects and let the user select one or more.

        mode can be set to'single' to select a single item.
        This sets the current selection to the selected names.
        Return a list with the selected names or None.
        """
        res = widgets.Selection(listAll(clas=self.clas),
                                'Known %sobjects' % self.object_type(),
                                mode,sort=True,selected=self.names
                                ).getResult()
        if res is not None:
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


    def printval(self):
        """Print the selection."""
        objects = self.check()
        if objects:
            for n,o in zip(self.names,objects):
                print "%s = %s" % (n,str(o))


    def printbbox(self):
        """Print the bbox of the current selection."""
        objects = self.check()
        if objects:
            for n,o in zip(self.names,objects):
                GD.message("Object %s has bbox %s" % (n,o.bbox()))
            if len(self.names) > 1:
                GD.message("Overal bbox is %s" % bbox(objects))



###################### Drawable Objects #############################

# Default Annotations
            
def draw_object_name(n):
    """Draw the name of an object at its center."""
    return drawText3D(named(n).center(),n)

def draw_elem_numbers(n):
    """Draw the numbers of an object's elements."""
    return drawNumbers(named(n))

def draw_bbox(n):
    """Draw the bbox of an object."""
    return drawBbox(named(n))
   

from gui.draw import *

class DrawableObjects(Objects):
    """A selection of drawable objects from the globals().

    annotations, if set, is a list of (func,active) tuples, where
    func is a function that is to be called with the object name as argument
      to draw some annotation for the object,
    active is a flag to signal if the annotation should be drawn or not.

    The default is to provide object name and element numbers.
    """
    def __init__(self,*args,**kargs):
        Objects.__init__(self,*args,**kargs)
        self.autodraw = False
        self.shrink = None
        self.annotations = [[draw_object_name,False],
                            [draw_elem_numbers,False],
                            [draw_bbox,False],
                            ]
        self._annotations = {}
        self._actors = []


    def draw(self,*args,**kargs):
        clear()
        GD.debug("Drawing SELECTION: %s" % self.names)
        self._actors = draw(self.names,clear=False,shrink=self.shrink,*args,**kargs)
        for i,a in enumerate(self.annotations):
            if a[1]:
                self.drawAnnotation(i)


    def ask(self,mode='multi'):
        """Interactively sets the current selection."""
        new = Objects.ask(self,mode)
        if new is not None:
            self.draw()


    def drawChanges(self):
        """Draws old and new version of a Formex with different colors.

        old and new can be a either Formex instances or names or lists thereof.
        old are drawn in yellow, new in the current color.
        """
        #print "DRAWCHANGES %s" % type(self)
        self.draw(wait=False)
        draw(self.values,color='yellow',bbox=None,clear=False,shrink=self.shrink)


    def undoChanges(self):
        """Undo the last changes of the values."""
        Objects.undoChanges(self)
        self.draw()


    def toggleAnnotation(self,i=0,onoff=None):
        """Toggle the display of number On or Off.

        If given, onoff is True or False. 
        If no onoff is given, this works as a toggle. 
        """
        active = self.annotations[i][1]
        #print "WAS"
        #print self.annotations
        if onoff is None:
            active = not active
        elif onoff:
            active = True
        else:
            active = False
        self.annotations[i][1] = active
        #print "BECOMES"
        #print self.annotations
        if active:
            self.drawAnnotation(i)
        else:
            self.removeAnnotation(i)
        #print self._annotations


    def drawAnnotation(self,i=0):
        """Draw some annotation for the current selection."""
        #print "DRAW %s" % i
        self._annotations[i] = [ self.annotations[i][0](n) for n in self.names ]


    def removeAnnotation(self,i=0):
        """Remove the annotation i."""
        #print "REMOVE %s" % i
        GD.canvas.removeAnnotations(self._annotations[i])
        GD.canvas.update()
        del self._annotations[i]


    def toggleNames(self):
        self.toggleAnnotation(0)
    def toggleNumbers(self):
        self.toggleAnnotation(1)
    def toggleBbox(self):
        self.toggleAnnotation(2)


    def setProperty(self,prop=None):
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


if __name__ == "draw":
    # If executed as a pyformex script
    GD.debug('Reloading module %s' % __file__)
    
elif __name__ == "__main__":
    print __doc__

# End

