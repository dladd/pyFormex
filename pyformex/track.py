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

"""track.py

Functions for creating classes with access tracking facilities.
This can e.g. be use to detect if the contents of a list or dict
has been changed.
"""

# List of methods to track. These are the methods that can possibly
# change the object. Objects of type 'list' and 'dict' are covered.
track_methods = [
    '__setitem__',
    '__setslice__',
    '__delitem__',
    'update',
    'append',
    'extend',
    'add',
    'insert',
    'pop',
    'popitem',
    'remove',
    'setdefault',
    '__iadd__'
    ]


def track_decorator(func):
    """Create a wrapper function for tracked class methods.
    
    The wrapper function increases the 'hits' attribute of the
    class and then executes the wrapped method.
    Note that the class is passed as the first argument.
    """
    def wrapper(*args, **kw):
        """Wrapper function for a class method."""
        #print "NOTIF %s %s" % (func.__name__, id(args[0]))
        args[0].hits += 1
        return func(*args, **kw)

    wrapper.__name__ = func.__name__
    return wrapper


def track_class_factory(cls,methods= track_methods):
    """Create a wrapper class with method tracking facilities.

    Given an input class, this will return a class with tracking
    facilities. The tracking occurs on all the specified methods.
    The default will track all methods which could possibly change
    a 'list' or a 'dict'.

    The class will get an extra attribute 'hits' counting the number
    of times one of these methods was called. This value can be reset to
    zero to track changes after some breakpoint.

    The methods should be owned by the class itself, not by a parent class.
    The default list of tracked method will track possible changes in
    'list' and 'dict' classes.

    Example:

      >>> TrackedDict = track_class_factory(dict)
      >>> D = TrackedDict({'a':1,'b':2})
      >>> print D.hits
      0
      >>> D['c'] = 3
      >>> print D.hits
      1
      >>> D.hits = 0
      >>> print D.hits
      0
      >>> D.update({'d':1,'b':3})
      >>> del D['a']
      >>> print D.hits
      2
    """
    new_dct = cls.__dict__.copy()
    if 'hits' in new_dct:
        raise ValueError,"The input class should not have an attribute 'hits'"
    
    for key, value in new_dct.items():
        if key in track_methods:
            new_value = track_decorator(value)
            new_dct[key] = new_value
    new_dct['hits'] = 0
    new_dct['__doc__'] = """Tracked %s class

""" % cls.__name__

    return type("track_"+ cls.__name__, (cls,), new_dct)


TrackedDict = track_class_factory(dict)
TrackedList = track_class_factory(list)

# End
