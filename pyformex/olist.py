# $Id$
##
##  This file is part of pyFormex 0.8.6  (Mon Jan 16 21:15:46 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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

"""Some convenient shortcuts for common list operations.

While most of these functions look (and work) like set operations, their
result differs from using Python builtin Sets in that they preserve the
order of the items in the lists.
"""
from __future__ import print_function

def roll(a,n=1):
    """Roll the elements of a list n positions forward (backward if n < 0)"""
    return a[n:] + a[:n]

def union(a,b):
    """Return a list with all items in a or in b, in the order of a,b."""
    return a + [ i for i in b if i not in a ]


def difference(a,b):
    """Return a list with all items in a but not in b, in the order of a."""
    return [ i for i in a if i not in b ]


def symdifference(a,b):
    """Return a list with all items in a or b but not in both."""
    return difference(a,b) + difference(b,a)


def intersection (a,b):
    """Return a list with all items in a and  in b, in the order of a."""
    return [ i for i in a if i in b ]


def concatenate(a):
    """Concatenate a list of lists"""
    return reduce(list.__add__,a)


def flatten(a,recurse=False):
    """Flatten a nested list.

    By default, lists are flattened one level deep.
    If recurse=True, flattening recurses through all sublists.

    >>> flatten([[[3.,2,],6.5,],[5],6,'hi'])
    [[3.0, 2], 6.5, 5, 6, 'hi']
    >>> flatten([[[3.,2,],6.5,],[5],6,'hi'],True)
    [3.0, 2, 6.5, 5, 6, 'hi']
    """
    r = []
    for i in a:
        if type(i) == list:
            if recurse:
                r.extend(flatten(i,True))
            else:
                r.extend(i)
        else:
            r.append(i)
    return r

    
def select(a,b):
    """Return a subset of items from a list.

    Returns a list with the items of a for which the index is in b.
    """
    return [ a[i] for i in b ]


def remove(a,b):
    """Returns the complement of select(a,b)."""
    return [ ai for i,ai in enumerate(a) if i not in b ]


def toFront(l,i):
    """Add or move i to the front of list l

    l is a list.
    If i is in the list, it is moved to the front of the list.
    Else i is added at the front of the list.

    This changes the list inplace and does not return a value.
    """
    if i in l:
        l.remove(i)
    l[0:0] = [ i ]
    
    


def collectOnLength(items,return_indices=False):
    """Collect items of a list in separate bins according to the item length.

    items is a list of items of any type having the len() method.
    The items are put in separate lists according to their length.

    The return value is a dict where the keys are item lengths and
    the values are lists of items with this length.

    If return_indices is True, a second dict is returned, with the same
    keys, holding the original indices of the items in the lists.
    """
    if return_indices:
        res,ind = {},{}
        for i,item in enumerate(items):
            li = len(item)
            if li in res.keys():
                res[li].append(item)
                ind[li].append(i)
            else:
                res[li] = [ item ]
                ind[li] = [ i ]
        return res,ind
    else:
        res = {}
        for item in items:
            li = len(item)
            if li in res.keys():
                res[li].append(item)
            else:
                res[li] = [ item ]
        return res


class List(list):
    def __init__(self,*args):
        list.__init__(self,*args)
    def __getattr__(self, attr):
        def on_all(*args, **kwargs):
            return List([ getattr(obj, attr)(*args, **kwargs) for obj in self ])
        return on_all


if __name__ == "__main__":

    a = [1,2,3,5,6,7]
    b = [2,3,4,7,8,9]
    print(a)
    print(b)
    print(union(a,b))
    print(difference(a,b))
    print(difference(b,a))
    print(symdifference(a,b))
    print(intersection(a,b))
    print(select(a,[1,3]))
    print(concatenate([a,b,a]))
    print(flatten([1,2,a,[a]]))
    print(flatten([1,2,a,[a]],recurse=True))
      


    class String(str):
        def __init__(self,s):
            str.__init__(s)
            self.length = len(s)
        def Len(self):
            return len(self)

    A = String("aa")
    B = String("bbbb")

    L = List([A,B])
    print(L.upper())

    print(L.Len())
    
# End
