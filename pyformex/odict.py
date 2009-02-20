#!/usr/bin/env python
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

"""A dictionary that keeps the keys in order of insertion."""


def listUnion(a,b):
    """Return a list with all items in a or in b, in the order of a,b."""
    return a + [ i for i in b if i not in a ]


def listDifference(a,b):
    """Return a list with all items in a but not in b, in the order of a."""
    return [ i for i in a if i not in b ]


def listSymDifference(a,b):
    """Return a list with all items in a or b but not in both."""
    return listDifference(a,b) + listDifference(b,a)


def listIntersection (a,b):
    """Return a list with all items in a and  in b, in the order of a."""
    return [ i for i in a if i in b ]

    
def listSelect(a,b):
    """Return a subset of items from a list.

    Returns a list with the items of a for which the index is in b.
    """
    return [ a[i] for i in b ]


def listConcatenate(a):
    """Concatenate a list of lists"""
    return reduce(list.__add__,a)


class ODict(dict):
    """An ordered dictionary.

    This is a dictionary that keeps the keys in order.
    The default order is the insertion order. The current order can be
    changed at any time.
    """

    def __init__(self,data={}):
        """Create a new ODict instance.

        The ODict can be initialized with a Python dict or an ODict.
        The order after insertion is indeterminate if a plain dict is used.
        """
        dict.__init__(self,data)
        if type(data) is ODict:
            self._order = data._order
        elif type(data) is list or type(data) is tuple:
            self._order = [ i[0] for i in data ]
        else:
            self._order = dict.keys(self)


    def __repr__(self):
        """Format the Dict as a string.

        We use the format Dict({}), so that the string is a valid Python
        representation of the Dict.
        """
        return [(k,self[k]) for k in self._order].__repr__()


    def __setitem__(self,key,value):
        """Allows items to be set using self[key] = value."""
        dict.__setitem__(self,key,value)
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)


    def __delitem__(self,key):
        """Allow items to be deleted using del self[key].

        Raises an error if key does not exist.
        """
        dict.__delitem__(self,key)
        self._order.remove(key)

        
    def update(self,data={}):
        """Add a dictionary to the ODict object.

        The new keys will be appended to the existing, but the order of the
        added keys is undetemined if data is a dict object. If data is an ODict
        its order will be respected.. 
        """
        dict.update(self,data)
        if type(data) is ODict:
            for k in data._order:
                if k in self._order:
                    self._order.remove(k)
            self._order += data._order


    def __add__(self,data):
        """Add two ODicts's together, returning the result."""
        self.update(data)
        return self


    def sort(self,keys):
        """Set the order of the keys.

        keys should be a list containing exactly all the keys from self.
        """
        if listDifference(keys,dict.keys(self)) != []:
            raise ValueError,"List of keys does not match current object's keys"
        self._order = keys


    def keys(self):
        """Return the keys in order."""
        return self._order


    def values(self):
        """Return the values in order of the keys."""
        return [self[k] for k in self._order]
    

    def items(self):
        """Return the key,value pairs in order of the keys."""
        return [(k,self[k]) for k in self._order]
    


class KeyList(ODict):
    """A named item list"""
    
    def __init__(self,alist=[]):
        """Create a new KeyList, possibly filling it with data.

        data should be a list of tuples/lists each having at
        least 2 elements.
        The (string value of the) first is used as the key.
        """
        L = map(len,alist)
        if min(L) < 2:
            raise ValueEror,"All items in the data should have length >= 2"
        ODict.__init__(self,[[i[0],i[1:]] for i in alist])
    

    def items(self):
        """Return the key+value lists in order of the keys."""
        return [(k,)+self[k] for k in self._order]

    
        

        

if __name__ == "__main__":

    a = [1,2,3,5,6,7]
    b = [2,3,4,7,8,9]
    print a
    print b
    print listUnion(a,b)
    print listDifference(a,b)
    print listDifference(b,a)
    print listIntersection(a,b)
    print a
    
    

    d = ODict({'a':1,'b':2,'c':3})
    print d
    d.sort(['a','b','c'])
    print d
    d['d'] = 4
    d['e'] = 5
    d['f'] = 6
    print d
    del d['c']
    print d
    D = ODict(d)
    print D
    D['d'] = 26
    print D
    print D['b']
    D = ODict(zip(range(5),range(6,10)))
    print D
    print D.keys()
    print D.values()
    print D.items()
    del D[1]
    del D[2]
    D[4] = 4
    D[3] = 3
    D[2] = 2
    D[1] = 1
    print D
    print D.keys()
    print D.values()
    print D.items()
    k = D.keys()
    k.sort()
    D.sort(k)
    print D
    print D.keys()
    print D.values()
    print D.items()
    
    
# End
