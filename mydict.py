#!/usr/bin/env python
"""
Extensions to Pythons built-in dictionary class:
Dict is a dictionary with default values and alternate attribute syntax.
CascadingDict is a Dict with lookup cascading into the next level Dict's
if the key is not found in the CascadingDict itself.

(C) 2005 Benedict Verhegghe
Distributed under the GNU GPL
"""

__all__ = [ 'Dict', 'CascadingDict' ]


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
        """Allows items to beset as self.key=value.

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



class CascadingKeyError(KeyError):
    """A KeyError exception for use in CascadingDict."""
    pass


class CascadingDict(Dict):
    """A cascading Dict: properties not in Dict are searched in all Dicts.

    This is equivalent to the Dict class, except that if a key is not found
    and the CascadingDict has items with values that are themselves instances
    of Dict or CascadingDict, the key will be looked up in those Dicts as well.

    As you expect, this will make the lookup cascade into all lower levels
    of CascadingDict's. The cascade will stop if you use a Dict.
    There is no way to guarantee in which order the (Cascading)Dict's are
    visited, so if multiple Dicts on the same level hold """


    def __init__(self,data={},default=None):
        Dict.__init__(self,data,default)


    def __getitem__(self, key):
        """Allows items to be addressed as self[key].

        This is equivalent to the Dict lookup, except that we
        cascade through lower level Dict's (not dict's though!).
        """
        #print "Lookin for ",key
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            try:
                for v in self.itervalues():
                    if isinstance(v,Dict):
                        #print "trying",v
                        try:
                            return v[key]
                        except CascadingKeyError:
                            #print "Pass"
                            pass
                raise CascadingKeyError
            except CascadingKeyError:
                pass
        return self.default
        
# this should be fixed in section instead
##    def getprop(self,key):
##    #nodig om 'getprop' op alle niveau's te doen werken (anders werkt bv section.getprop('A') niet)
##	return self. __getattr__(key)


if __name__ == '__main__':

    C = Dict({'a':1,'x':Dict({'b':'2','x':Dict({'c':3,'x':Dict({'d':4,'a':0})})})})
    print C
    print C.a,C.b,C.c,C.d,C.e,C.x.a # This only finds C.a : no cascading

    C = CascadingDict(C)
    print C.a,C.b,C.c,C.d,C.e,C.x.a # This finds C.a and C.b: 1 level cascading

    C = CascadingDict({'a':1,'x':CascadingDict({'b':'2','x':CascadingDict({'c':3,'x':CascadingDict({'d':4,'a':0})})})})
    print C.a,C.b,C.c,C.d,C.e,C.x.a # This finds everything except C.e

    C = CascadingDict({'a':1,'x':CascadingDict({'b':'2','x':CascadingDict({'c':3,'x':Dict({'d':4,'a':0})})})})
    print C.a,C.b,C.c,C.d,C.e,C.x.a # Same here, forelast level is still cascading

    C = CascadingDict({'a':1,'x':CascadingDict({'b':'2','x':CascadingDict({'c':3,'x':dict({'d':4,'a':0})})})})
    print C.a,C.b,C.c,C.d,C.e,C.x.a # Does not find d, x.a because last level is a dict, not Dict

    C = CascadingDict({'a':1,'x':CascadingDict({'b':'2','x':Dict({'c':3,'x':Dict({'d':4,'a':0})})})})
    print C.a,C.b,C.c,C.d,C.e,C.x.a # Does not find d, x.a because forelast level is not cascading

