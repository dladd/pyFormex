#!/usr/bin/env python
# $Id$

"""olist.py: Some convenient shortcuts for common list operations.

While most of these functions look (and work) like set operations, their
result differs from using Python builtin Sets in that they preserve the
order of the items in the lists.
"""


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

    
def select(a,b):
    """Return a subset of items from a list.

    Returns a list with the items of a for which the index is in b.
    """
    return [ a[i] for i in b ]
        

if __name__ == "__main__":

    a = [1,2,3,5,6,7]
    b = [2,3,4,7,8,9]
    print a
    print b
    print union(a,b)
    print difference(a,b)
    print difference(b,a)
    print symdifference(a,b)
    print intersection(a,b)
    print select(a,[1,3])
    print concatenate([a,b,a])
      
    
# End
