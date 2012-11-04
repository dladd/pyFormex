#!/usr/bin/python
# $Id$
##
##  This file is part of pyFormex 0.8.8  (Sun Nov  4 17:22:49 CET 2012)
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

"""Specialized dictionary type structures.

"""
from __future__ import print_function

## import utils
## have_version = utils.hasModule('python')
## if utils.SaneVersion(have_version) < utils.SaneVersion('2.7'):
##     from backports import OrderedDict
## else:
##     from collections import OrderedDict

    
import olist


def __newobj__(cls, *args):
    return cls.__new__(cls, *args)


class ODict(dict):
    """**An ordered dictionary.**

    This is a dictionary that keeps the keys in order.
    The default order is the insertion order. The current order can be
    changed at any time.

    The :class:`ODict` can be initialized with a Python dict, a list of
    (key,value) tuples, or another     :class:`ODict` object. If a plain Python dict is used, the resulting
    order is undefined.
    """
    def __init__(self,data={},):
        """Create a new ODict instance."""
        dict.__init__(self,data)

        if isinstance(data,ODict):
            # keep order
            self._order = data._order
            
        elif type(data) is list or type(data) is tuple:
            # preserve the order
            self._order = []
            self._add_keys([i[0] for i in data])
            
        elif type(data) is dict:
            # order is undefined
            self._order = data.keys()

        else:
            raise ValueError,"Unexpected initialization value for ODict"
  

    def _add_keys(self,keys):
        """Add a list of keys to the ordered list, removing existing keys."""
        for k in keys:
            if k in self._order:
                self._order.remove(k)
        self._order += keys
        
        
    def update(self,data={}):
        """Add a dictionary to the ODict object.

        The new keys will be appended to the existing, but the order of the
        added keys is undetemined if data is a dict object. If data is an ODict
        its order will be respected.. 
        """
        dict.update(self,data)
        self._add_keys(ODict(data)._order)


    def __iter__(self):
        return list.__iter__(self._order)


    def __repr__(self):
        """Format the Dict as a string.

        We use the format Dict({}), so that the string is a valid Python
        representation of the Dict.
        """
        return [(k,self[k]) for k in self._order].__repr__()


    def __setitem__(self,key,value):
        """Allows items to be set using self[key] = value."""
        dict.__setitem__(self,key,value)
# setting an item should not change the order!
# If you want to change the order, first remove the item
##         if key in self._order:
##             self._order.remove(key)
##         self._order.append(key)
        if key not in self._order:
            self._order.append(key)


    def __delitem__(self,key):
        """Allow items to be deleted using del self[key].

        Raises an error if key does not exist.
        """
        dict.__delitem__(self,key)
        self._order.remove(key)


    def __add__(self,data):
        """Add two ODicts's together, returning the result."""
        self.update(data)
        return self


    def sort(self,keys):
        """Set the order of the keys.

        keys should be a list containing exactly all the keys from self.
        """
        if olist.difference(keys,dict.keys(self)) != []:
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


    def pos(self,key):
        """Return the position of the specified key.

        If the key is not in the ODict, None is returned"""
        try:
            return self._order.index(key)
        except ValueError:
            return None
        

    def __reduce__(self):
        state = (dict(self), self.__dict__)
        return (__newobj__, (self.__class__,), state)


    def __setstate__(self,state):
        self.__init__()
        if type(state) == tuple:
            self.update(state[0])
            self.__dict__.update(state[1])
        elif type(state) == dict:
            #self.__dict__['_default_'] = state.pop('_default_')
            self.update(state)



class KeyedList(ODict):
    """A named item list.

    A KeyedList is a list of lists or tuples. Each item (sublist or tuple)
    should at least have 2 elements: the first one is used as a key to
    identify the item, but is also part of the information (value) of the
    item.
    """
    
    def __init__(self,alist=[]):
        """Create a new KeyedList, possibly filling it with data.

        data should be a list of tuples/lists each having at
        least 2 elements.
        The (string value of the) first is used as the key.
        """
        L = map(len,alist)
        if min(L) < 2:
            raise ValueEror,"All items in the data should have length >= 2"
        ODict.__init__(self,[[i[0],i[1:]] for i in alist])
        print(self)
    

    def items(self):
        """Return the key+value lists in order of the keys."""
        return [(k,)+self[k] for k in self._order]


if __name__ == "__main__":

    d = ODict({'a':1,'b':2,'c':3,'a':3})
    print(d)
    d.sort(['a','b','c'])
    print(d)
    d = ODict([('a',1),('b',2),('c',3),('a',3)])
    print(d)
    d['d'] = 4
    d['e'] = 5
    d['f'] = 6
    print(d)
    del d['c']
    print(d)
    D = ODict(d)
    print(D)
    D['d'] = 26
    print(D)
    print(D['b'])
    D = ODict(zip(range(5),range(6,10)))
    print(D)
    print(D.keys())
    print(D.values())
    print(D.items())
    del D[1]
    del D[2]
    D[4] = 4
    D[3] = 3
    D[2] = 2
    D[1] = 1
    print(D)
    print(D.keys())
    print(D.values())
    print(D.items())
    k = D.keys()
    k.sort()
    D.sort(k)
    print(D)
    print(D.keys())
    print(D.values())
    print(D.items())

    D[1] += 7
    D[3] += 8

    print(D.items())

    print("DONE")
    
# End
