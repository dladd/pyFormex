#!/usr/bin/env python pyformex.py
# $Id$

"""objects.py

This is a support module for other pyFormex plugins.
"""

import globaldata as GD
from formex import Formex
from plugins.surface import Surface
from gui.draw import *


class Objects(object):
    """A selection of objects from the globals().

    The class provides facilities to filter the global objects by their type
    and select one or more objects by their name(s). The values of these
    objects can be changed and the changes can be undone.
    """

    def __init__(self,dic=None,clas=None,filter=None,namelist=[]):
        """Create a new selection of objects.

        If a dict is given, objects will be selected from this dict, else
        from the global pyFormex dict GD.PF.
        If a filter is given, only objects passing it will be accepted.
        The filter will be applied dynamically on the dict.

        If a list of names is given, the current selection will be set to
        those names (provided they are in the dictionary.
        """
        if dict is None:
            self.dic = GD.PF
        else:
            self.dic = dic
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


    def clear(self):
        """Clear the selection."""
        self.names = []
        self.values = []
        

    def __getitem__(self,i):
        """Return selection item i"""
        return self.names[i]
    

    def selectAll(self):
        self.set(self.listAll(self.clas))


    def remember(self):
        self.values = map(named,self.names)
        print self.values
        

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
        an error message is displayed and an empty list is returned.
        """
        if len(self.names) == 0:
            if warn:
                warning("No %sobjects were selected" % self.object_type())
            return []
        if single and len(self.names) > 1:
            if warn:
                warning("You should select exactly one %sobject" %  self.object_type())
            return []
        if single:
            return named(self.names[0])
        else:
            return map(named,self.names)


    def ask(self,mode='multi'):
        """Show the names of known objects and let the user select one or more.

        mode can be set to'single' to select a single item.
        This sets the current selection to the selected names.
        Return the selected names or None.
        """
        res = widgets.Selection(listAll(clas=self.clas),
                                'Known %sobjects' % self.object_type(),
                                mode,sort=True,selected=self.names
                                ).getResult()
        if res is not None:
            self.set(res)
        return res


    def forget(self):
        """Remove the selection from the globals."""
        forget(self.names)
        self.clear()


class DrawableObjects(Objects):
    """A selection of drawable objects from the globals().

    """
    def __init__(self,*args,**kargs):
        Objects.__init__(self,*args,**kargs)
        self.autodraw = False
        self.show_vert_numbers = False
        self.show_edge_numbers = False
        self.show_elem_numbers = False
        self.shrink = None


    def draw(self,*args,**kargs):
        clear()
        print "DRAWING WITH SHRINK = %s" % self.shrink
        draw(self.names,clear=False,shrink=self.shrink,*args,**kargs)
        #if show_numbers:
        #    showNumbers()


    def ask(self,mode='multi'):
        """Interactively sets the current selection."""
        new = Objects.ask(self,mode)
        if new is not None:
            self.draw()


    def drawChanges(self):
        """Draws old and new version of a Formex with differrent colors.

        old and new can be a either Formex instances or names or lists thereof.
        old are drawn in yellow, new in the current color.
        """
        self.draw(wait=False)
        draw(self.values,color='yellow',bbox=None,clear=False,shrink=self.shrink)
 

    def undoChanges(self):
        """Undo the last changes of the values."""
        Objects.undoChanges(self)
        self.draw()
       

if __name__ == "__main__":
    print __doc__

# End

