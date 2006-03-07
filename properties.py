#!/usr/bin/env python
# $Id$
"""General framework for attributing properties to Formex elements.

Properties can really be just about any Python object.
Properties are identified and connected to a Formex element by the
prop values that are stored in the Formex.
"""

from mydict import CascadingDict


class Property(CascadingDict):
    """A general properties class.

    This class should only provide general methods, such as
    add, change and delete properties, lookup, print, and
    of course, connect properties to Formex elements.
    """

    def __init__(self,data={},default=None):
        """Create a new property. Empty by default."""
        CascadingDict.__init__(self,data,default)


ls TeX

    def __repr__(self):
        """Format a property into a string."""
        s = "PropertyClass{ default=%s" % self.default
        for i in self.items():
            s += "\n  %s = %s" % i
        return s + "}\n"


# Test

if __name__ == "__main__":

    cload1 = {'x':1.0, 'z':1.0, 'ry':0.0 }
    cload2 = {'x':2.0, 'y':3.0 } 
    bdisp = {'x':0.0, 'rx':0.0, 'rz':0.0}

    np1 = Property({'cload':cload1,'bound':bdisp})
    np2 = Property({'cload':cload2,'bound':bdisp,'color':'red','id':'123'})

    np = {}
    np['0'] = np1
    np['1'] = np2
    np['2'] = np1

    for i in range(3):
        print np[str(i)]
