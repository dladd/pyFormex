#!/usr/bin/env python
# $Id$
##
##  This file is part of pyFormex 0.8.1 Release Tue Dec  8 12:25:08 2009
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Homepage: http://pyformex.org   (http://pyformex.berlios.de)
##  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
"""
Extensions to Pythons built-in dictionary class:
Dict is a dictionary with default values and alternate attribute syntax.
CDict is a Dict with lookup cascading into the next level Dict's
if the key is not found in the CDict itself.

(C) 2005,2008 Benedict Verhegghe
Distributed under the GNU GPL version 3 or later
"""

import copy


def cascade(d, key):
    """Cascading lookup in a dictionary.

    This is equivalent to the dict lookup, except that when the key is
    not found, a cascading lookup through lower level dict's is started
    and the first matching key found is returned.
    """
    try:
        return dict.__getitem__(d,key)
    except KeyError:
        for v in d.itervalues():
            if isinstance(v,dict):
                try:
                    return cascade(v,key)
                except KeyError:
                    pass
        raise KeyError



def returnNone(key):
    """Always returns None."""
    return None

def raiseKeyError(key):
    """Raise a KeyError."""
    raise KeyError,"Not found: %s" % key


class Dict(dict):
    """A Python dictionary with default values and attribute syntax.

    :class:`Dict` is functionally nearly equivalent with the builtin Python
    :class:`dict`, but provides the following extras:
    
    - Items can be accessed with attribute syntax as well as dictionary
      syntax. Thus, if ``C`` is a :class:`Dict`, ``C['foo']`` and ``C.foo``
      are equivalent.
      This works as well for accessing values as for setting values.
      In the following, the terms *key* or *attribute* therefore have the
      same meaning.
    - Lookup of a nonexisting key/attribute does not automatically raise an
      error, but calls a ``_default_`` lookup method which can be set by
      the user.
      The default is to raise a KeyError, but an alternative is to return
      None or some other default value.

    There are a few caveats though:
    
    - Keys that are also attributes of the builtin dict type, can not be used
      with the attribute syntax to get values from the Dict. You should use
      the dictionary syntax to access these items. It is possible to set
      such keys as attributes. Thus the following will work::
      
         C['get'] = 'foo'
         C.get = 'foo'
         print(C['get'])
         
      but this will not::
      
         print(C.get)

      This is done so because we want all the dict attributes to be available
      with their normal binding. Thus, ::
      
         print(C.get('get'))

      will print ``foo``
    

    To avoid name clashes with user defines, many Python internal names start
    and end with '__'. The user should avoid such names.
    The Python dict has the following attributes not enclosed between '__',
    so these are the ones to watch out for:
    'clear', 'copy', 'fromkeys', 'get', 'has_key', 'items', 'iteritems',
    'iterkeys', 'itervalues', 'keys', 'pop', 'popitem', 'setdefault',
    'update', 'values'.
    """

    def __init__(self,data={},default=None):
        """Create a new Dict instance.

        The Dict can be initialized with a Python dict or a Dict.
        If defined, default is a function that is used for alternate key
        lookup if the key was not found in the dict.
        """
        dict.__init__(self,data.items())
        if default is None:
            default = raiseKeyError
        if not callable(default):
            raise ValueError,"'default' should be a callable function" 
        self.__dict__['_default_'] = default


    def __repr__(self):
        """Format the Dict as a string.

        We use the format Dict({}), so that the string is a valid Python
        representation of the Dict.
        """
        return "Dict(%s)" % dict.__repr__(self)


    def __getitem__(self, key):
        """Allows items to be addressed as self[key].

        This is equivalent to the dict lookup, except that we
        provide a default value if the key does not exist.
        """
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            return self._default_(key)


    def __delitem__(self,key):
        """Allow items to be deleted using del self[key].

        Silently ignore if key is nonexistant.
        """
        try:
            dict.__delitem__(self,key)
        except KeyError:
            pass


    def __getattr__(self,key):
        """Allows items to be addressed as self.key.

        This makes self.key equivalent to self['key'], except if key
        is an attribute of the builtin type 'dict': then we return that
        attribute instead, so that the 'dict' methods keep their binding.
        """
        try:
            return dict.__getattribute__(self,key)
        except AttributeError:
            if key == '_default_':
                return self.__dict__['_default_']
            else:
                return self.__getitem__(key)


    def __setattr__(self,key,value=None):
        """Allows items to be set as self.key=value.

        This works even if the key is an existing attribute of the
        builtin dict class: the key,value pair is stored in the dict,
        leaving the dict's attributes unchanged.
        """
        self.__setitem__(key,value)


    def __delattr__(self,key):
        """Allow items to be deleted using del self.key.

        This works even if the key is an existing attribute of the
        builtin dict class: the item is deleted from the dict,
        leaving the dict's attributes unchanged.
        """
        self.__delitem__(key)


    def update(self,data={}):
        """Add a dictionary to the Dict object.

        The data can be a dict or Dict type object. 
        """
        dict.update(self,data)


    def get(self, key, default):
        """Return the value for key or a default.

        This is the equivalent of the dict get method, except that it
        returns only the default value if the key was not found in self,
        and there is no _default_ method or it raised a KeyError.
        """
        try:
            return self[key]
        except KeyError:
            return default

    # Added this to keep pydoc happy. Probably we should redefine
    # this one instead of get?
    #__get__ = get
    ###   !!!!!!!!  We had to remove it to keep the functionality
    ###   of the module!!!!!

    def setdefault(self, key, default):
        """Replaces the setdefault function of a normal dictionary.

        This is the same as the get method, except that it also sets the
        default value if get found a KeyError.
        """
        try:
            return self[key]
        except KeyError:
            self[key] = default
            return default


    def __deepcopy__(self,memo):
        """Create a deep copy of ourself."""
        newdict = self.__class__(default=self._default_)
        for k,v in self.items():
            newdict[k] = copy.deepcopy(v,memo)
        return newdict


##    def __getstate__(self):
##        d = copy.copy(self.__dict__)
##        d.update(self)
##        return d


##    def __setstate__(self,d):
##        self.__dict__['_default_'] = d.pop('_default_')
##        self.update(d)


    def __reduce__(self):
        state = (dict(self), self.__dict__)
        return (__newobj__, (self.__class__,), state)


    def __setstate__(self,state):
        if type(state) == tuple:
            self.update(state[0])
            self.__dict__.update(state[1])
        elif type(state) == dict:
            self.__dict__['_default_'] = state.pop('_default_')
            self.update(state)


def __newobj__(cls, *args):
    return cls.__new__(cls, *args)


_indent = 0  # number of spaces to indent in __str__ formatting
             # incremented by 2 on each level

class CDict(Dict):
    """A cascading Dict: properties not in Dict are searched in all Dicts.

    This is equivalent to the Dict class, except that if a key is not found
    and the CDict has items with values that are themselves instances
    of Dict or CDict, the key will be looked up in those Dicts as well.

    As you expect, this will make the lookup cascade into all lower levels
    of CDict's. The cascade will stop if you use a Dict.
    There is no way to guarantee in which order the (Cascading)Dict's are
    visited, so if multiple Dicts on the same level hold the same key,
    you should know yourself what you are doing.
    """


    def __init__(self,data={},default=returnNone):
        Dict.__init__(self,data,default)


    def __repr__(self):
        """Format the CDict as a string.

        We use the format Dict({}), so that the string is a valid Python
        representation of the Dict.
        """
        return "CDict(%s)" % dict.__repr__(self)

    
    def __str__(self):
        """Format a CDict into a string."""
        global _indent
        s = ""
        _indent += 2
        for i in self.items():
            s += '\n' + (' '*_indent) + "%s = %s" % i
        _indent -= 2
        return s


    def __getitem__(self, key):
        """Allows items to be addressed as self[key].

        This is equivalent to the dict lookup, except that we
        cascade through lower level dict's.
        """
        try:
            return cascade(self, key)
        except KeyError:
            return self._default_(key)


if __name__ == '__main__':

    import cPickle as pickle
    global C,Cr,Cs

    def val(s,typ='s'):
        """Returns a string assigning the value of s to the name s."""
        try:
            return ("%s = %"+typ) % (s,eval(s))
        except:
            return "Error in %s" % s

    def show():
        """Print C with '%r' and '%s' formatting."""
        global C,Cr,Cs
        Cr = val('C','r')
        Cs = val('C','s')
        print(Cr)
        print(Cs)
        print(C.get('y','yorro'))
        print(C.get('z','zorro'))
        print(C.setdefault('w','worro'))
        print("%s = %s" % (C['y']['c'],C.y.c))


    def testpickle():
        global C
        print("Test (un)pickle")
        f = file('test.pickle','w')
        print(type(C))
        print(C._default_)
        pickle.dump(C,f)
        f.close()
        f = file('test.pickle','r')
        C = pickle.load(f)
        print(type(C))
        print(C._default_)
        f.close()
        

    C = Dict({'x':Dict({'a':1,'y':Dict({'b':5,'c':6})}),'y':Dict({'c':3,'d':4}),'d':0})
    show()
    testpickle()
    show()
    
    # now exec this to check if we get the same
    exec(Cr)
    show()

    # now replace Dict with CDict
    Cr = Cr.replace('Dict','CDict')
    exec(Cr)
    show()
    testpickle()
    show()

    # show some items
    print(val("C['a'],C['b'],C['c'],C['d'],C['x']['c']"))
    print(val("C['e']"))
    print(val("C.a,C.b,C.c,C.d,C.x.c"))
    print(val("C.e"))
    

    C = CDict({'a':'aa','d':{'a':'aaa','b':'bbb'}})
    print(C)
    print(C['a'])
    print(C['d'])
    print(C['b'])

    D = copy.deepcopy(C)
    print(D)
    print(D.__dict__)
    print("%s == %s" % (C['b'],D['b']) )

    C = Dict({'a':'aa','d':{'a':'aaa','b':'bbb'}})
    print(C)
    print(C['a'])
    print(C['d'])
    try:
        print(C['b'])
        print("This should provoke a KeyError, so you should not see this text")
    except:
        print("Correctly received the intended KeyError")

    D = copy.deepcopy(C)
    print(D)
    print(D.__dict__)

# End

