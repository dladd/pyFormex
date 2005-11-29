#!/usr/bin/env python
# $Id$
"""Definition of some typical property sets used in structural analysis.

This is currently mainly for testing different implementations.
The classes in this module should not be relied upon. They may
dissappear in future.
"""

class NodeProperty:
    """Properties related to a single node."""

    def __init__(self,cload=None,bound=None):
        """Create a new node property. Empty by default

        A node property can hold the following sub-properties:
        - cload : a concentrated load
        - bound : a boundary condition
        """
        self.cload = cload
        self.bound = bound

    def __repr__(self):
        """Format a node property into a string."""
        return "{ cload: %s; bound: %s }" % (str(self.cload),str(self.bound))

    def __getitem__(self,name):
        """Return a named attribute."""
        return getattr(self,name)


from UserDict import UserDict
class AltNodeProperty(UserDict):
    """An alternate node property class."""

    def __init__(self,init={}):
        """Create a new node property. Empty by default

        A node property can hold anything in dictionary format
        """
        self.data = init

    def __getitem__(self,name):
        if self.has_key(name):
            return self.data[name]
        else:
            return None

    def __getattr__(self,name):
        if self.has_key(name):
            return self[name]
        else:
            raise AttributeError

# Test

if __name__ == "__main__":

    P1 = [ 1.0,1.0,1.0, 0.0,0.0,0.0 ]
    P2 = [ 0.0 ] * 3 + [ 1.0 ] * 3 
    B1 = [ 0.0 ] * 6

    np = {}
    np['1'] = NodeProperty(P1)
    np['2'] = NodeProperty(cload=P2)
    np['3'] = np['2']
    np['3'].bound = B1
    np['1'].cload[1] = 33.0

    np['4'] = AltNodeProperty({'cload':P1})
    np['5'] = AltNodeProperty({'cload':P2})
    np['6'] = np['5']
    np['6']['bound'] = B1
    np['4']['cload'][1] = 33.0

    np7 = NodeProperty(bound=B1)
    np8 = AltNodeProperty({'bound':B1})
    np['7'] = np7
    np['8'] = np8

    for key,item in np.iteritems():
        print key,item

    print "'cload' attributes"
    for key,item in np.iteritems():
        print key,item['cload']

    print "cload attributes"
    for key,item in np.iteritems():
        print key,item.cload
