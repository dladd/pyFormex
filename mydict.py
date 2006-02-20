#!/usr/bin/env python
"""
A dictionary with default values and attribute syntax.

(C) 2005 Benedict Verhegghe
Distributed under the GNU GPL
"""


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


##    def __setitem__(self,key,value=None):
##        """Allows items to be set as self[key]=value"""
##        dict.__setitem__(self,key,value)


    def __delitem__(self,key):
        """Allow items to be deleted using del self[key].

        Silently ignore if key is nonexistant.
        """
        try:
            dict.__delitem__(self, key)
        except KeyError:
            pass


##    def __getattr__(self, key):
##        """Allows items to be addressed as self.key."""
##        try:
##            return self.__getattribute__(key)
##        except AttributeError:
##            return None


    def __getattr__(self, key):
        """Allows items to be addressed as self.key.

        This make self.key become equivalent to self['key'], except if key
        is an attribute of the builtin type dict: then we return that
        attribute, so that the dict methods keep their binding.
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


##    def __str__(self):
##        return str(self.__dict__)



if __name__ == '__main__':

    C = Dict({'a':1,'b':2})
    print C
    print C.a
    print C.b
    try:
        print C.c
    except:
        print "Can not print C.c"
    print C['a']
    print C['b']
    print C['c']
    C.a = 3
    C['b'] = 4
    C['d'] = 7
    C.d = 8
    C.update({'b':5,'c':6})
    print C
    print C.__dict__
    print C.get('hallo')
    print C.get('hallo',24)
    C.get = 123
    print C
    print C.__dict__
    print C.get('hallo')
    print C.get('get')
    print C.keys()
    print C.items()
    print dict(C.items())
    print C.get
    
    D=Dict(C)
    D.a = None
    del D['b']
    # del D.b # does not work
    print D
    print C
    print C.__dict__
    print len(C)
    del D['aaa']
    del D.b
    del D.get
    print D.get('d')
    print D
    D.update(C)
    print D
