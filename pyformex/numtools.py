#!/usr/bin/env python
# $Id$
##
## This file is part of pyFormex 0.7.2 Release Tue Sep 23 16:18:43 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""numtools.py: Our additions to the numpy classes.

This is a collection of definitions that depend on the numpy module.
"""

import pyformex as GD
from numpy import *

################# Collection of Actors or Actor Elements ###############

class Collection(object):
    """A collection  is a set of (int,int) tuples.

    The first part of the tuple has a limited number of values and are used
    as the keys in a dict.
    The second part can have a lot of different values and is implemented
    as an integer array with unique values.
    This is e.g. used to identify a set of individual parts of one or more
    OpenGL actors.
    """
    def __init__(self):
        self.d = {}
        self.obj_type = None

    def setType(self,obj_type):
        self.obj_type = obj_type

    def clear(self,keys=[]):
        if keys:
            for k in keys:
                k = int(k)
                if k in self.d.keys():
                    del self.d[k]
        else:
            self.d = {}

    def add(self,data,key=-1):
        """Add new data to the collection.

        data can be a 2d array with (key,val) tuples or a 1-d array
        of values. In the latter case, the key has to be specified
        separately, or a default value will be used.
        """
        if len(data) == 0:
            return
        data = asarray(data)
        if data.ndim == 2:
            for key in unique1d(data[:,0]):
                self.add(data[data[:,0]==key,1],key)

        else:
            key = int(key)
            data = unique1d(data)
            if self.d.has_key(key):
                self.d[key] = union1d(self.d[key],data)
            elif data.size > 0:
                self.d[key] = data

    def set(self,data,key=-1):
        """Set the collection to the specified data.

        This is equivalent to clearing the corresponding keys
        before adding.
        """
        self.clear()
        self.add(data,key)

    def remove(self,data,key=-1):
        """Remove data from the collection."""
        data = asarray(data)
        if data.ndim == 2:
            for key in unique1d(data[:,0]):
                self.remove(data[data[:,0]==key,1],key)

        else:
            key = int(key)
            if self.d.has_key(key):
                data = setdiff1d(self.d[key],unique1d(data))
                if data.size > 0:
                    self.d[key] = data
                else:
                    del self.d[key]
            else:
                GD.debug("Not removing from non-existing selection for actor %s" % key)
    
    def has_key(self,key):
        """Check whether the collection has an entry for the key."""
        return self.d.has_key(key)

    def __setitem__(self,key,data):
        """Set new values for the given key."""
        key = int(key)
        data = unique1d(data)
        if data.size > 0:
            self.d[key] = data
        else:
            del self.d[key]

    def __getitem__(self,key):
        """Return item with given key."""
        return self.d[key]

    def get(self,key,default=[]):
        """Return item with given key or default."""
        key = int(key)
        return self.d.get(key,default)


    def keys(self):
        """Return a sorted array with the keys"""
        k = asarray(self.d.keys())
        k.sort()
        return k
        
        
    def __str__(self):
        s = ''
        keys = self.d.keys()
        keys.sort()
        for k in keys:
            s += "%s %s; " % (k,self.d[k])
        return s


################# Testing ###############

if __name__ == "__main__":
    print "Testing the Collection object"
    a = Collection()
    a.add(range(7),3)
    a.add(range(4))
    a.remove([2,4],3)
    print a
    a.add([[2,0],[2,3],[-1,7],[3,88]])
    print a
    a[2] = [1,2,3]
    print a
    a[2] = []
    print a
    a.set([[2,0],[2,3],[-1,7],[3,88]])
    print a

                    
# End
