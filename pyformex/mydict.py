#!/usr/bin/env python
##
## This file is part of pyFormex 0.7 Release Fri Apr  4 18:41:11 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""
Extensions to Pythons built-in dictionary class:
Dict is a dictionary with default values and alternate attribute syntax.
CascadingDict is a Dict with lookup cascading into the next level Dict's
if the key is not found in the CascadingDict itself.

(C) 2005 Benedict Verhegghe
Distributed under the GNU GPL
"""

__all__ = [ 'Dict', 'CascadingDict', 'cascade' ]


def cascade(dic, key):
    """Cascading lookup in a dictionary.

    This is equivalent to the dict lookup, except that when the key is
    not found, a cascading lookup through lower level dict's is started
    and the first matching key found is returned.
    """
    try:
        return dict.__getitem__(dic,key)
    except KeyError:
        for v in dic.itervalues():
            if isinstance(v,dict):
                try:
                    return cascade(v,key)
                except KeyError:
                    pass
        raise KeyError


class Dict(dict):
    """A Python dictionary with default values and attribute syntax.

    Dict is functionally nearly equivalent with the builtin Python dict,
    but provides the following extras:
    - Items can be accessed with attribute syntax as well as dictionary
      syntax. Thus, if C is a Dict, the following are equivalent:
          C['foo']   or   C.foo
      This works as well for accessing values as for setting values.
      In the following, the words key or attribute therefore have the
      same meaning.
    - Lookup of a nonexisting key/attribute does not raise an error, but
      returns a default value which can be set by the user (None by default).

    There are a few caveats though:
    - Keys that are also attributes of the builtin dict type, can not be used
      with the attribute syntax to get values from the Dict. You should use
      the dictionary syntax to access these items. It is possible to set
      such keys as attributes. Thus the following will work:
         C['get'] = 'foo'
         C.get = 'foo'
         print C['get']
      but not
         print C.get

      This is done so because we want all the dict attributes to be available
      with their normal binding. Thus,
         print C.get('get')
      will print
         foo
      
    """


    def __init__(self,data={},default=None):
        """Create a new Dict instance.

        The Dict can be initialized with a Python dict or a Dict.
        The default value is the value that will be returned on lookups
        for non-existing keys.
        """
        dict.__init__(self,data.items())
        # Do not use self.default= here, because the __setattr__
        # method would put it in the base class dict 
        self.__dict__['default'] = default


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
            return self.default


    def __delitem__(self,key):
        """Allow items to be deleted using del self[key].

        Silently ignore if key is nonexistant.
        """
        try:
            dict.__delitem__(self, key)
        except KeyError:
            pass


    def __getattr__(self, key):
        """Allows items to be addressed as self.key.

        This makes self.key equivalent to self['key'], except if key
        is an attribute of the builtin type 'dict': then we return that
        attribute instead, so that the 'dict' methods keep their binding.
        """
        try:
            return dict.__getattribute__(self,key)
        except AttributeError:
            return self.__getitem__(key)


    def __setattr__(self, key,value=None):
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


_indent = 0  # number of spaces to indent in __str__ formatting
             # incremented by 2 on each level

class CascadingDict(Dict):
    """A cascading Dict: properties not in Dict are searched in all Dicts.

    This is equivalent to the Dict class, except that if a key is not found
    and the CascadingDict has items with values that are themselves instances
    of Dict or CascadingDict, the key will be looked up in those Dicts as well.

    As you expect, this will make the lookup cascade into all lower levels
    of CascadingDict's. The cascade will stop if you use a Dict.
    There is no way to guarantee in which order the (Cascading)Dict's are
    visited, so if multiple Dicts on the same level hold the same key,
    you should know yourself what you are doing.
    """


    def __init__(self,data={},default=None):
        Dict.__init__(self,data,default)


    def __repr__(self):
        """Format the CascadingDict as a string.

        We use the format Dict({}), so that the string is a valid Python
        representation of the Dict.
        """
        return "CascadingDict(%s)" % dict.__repr__(self)

    
    def __str__(self):
        """Format a CascadingDict into a string."""
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
            return self.default


if __name__ == '__main__':

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
        print Cr
        print Cs
        

    C = Dict({'x':Dict({'a':1,'y':Dict({'b':5,'c':6})}),'y':Dict({'c':3,'d':4}),'d':0})
    show()
    # now exec this to check if we get the same
    exec(Cr)
    show()
    # now replace Dict with CascadingDict
    Cr = Cr.replace('Dict','CascadingDict')
    exec(Cr)
    show()
    # show some items
    print val("C['a'],C['b'],C['c'],C['d'],C['x']['c']")
    print val("C['e']")
    print val("C.a,C.b,C.c,C.d,C.x.c")
    print val("C.e")
    
