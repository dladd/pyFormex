#!/usr/bin/env python
"""
A general yet simple configuration class.

(C) 2005 Benedict Verhegghe
Distributed under the GNU GPL

Why:
I wrote this simple class because I wanted to use Python expressions in my
configuration files. This is so much more fun than using .INI style config
files.
While there are some other Python config modules available on the web,
I couldn't find one that suited my needs and my taste: either they are intended
for more complex configuration needs than mine, or they do not work with the
simple Python syntax I expected.

What:
Our Config class is just a normal Python dictionary which can hold anything.
Fields can be accessed either as dictionary lookup (config['foo']) or as
object attributes (config.foo).
The class provides a function for reading the dictionary from a flat text
(multiline string or file). I will always use the word 'file' hereafter,
because that is what you usually will read the configuration from.
Your configuration file can have named sections. Sections are stored as
other Python dicts inside the top Config dictionary. The current version
is limited to one level of sectioning.
"""

import copy

def dicttostr(dic):
    """Format a dict in Python source representation.

    Each (key,value) pair is formatted on a line : key = value.
    """
    s = ""
    if isinstance(dic,dict):
        for k,v in dic.iteritems():
            if type(v) == str:
                s += '%s = "%s"\n' % (k,v)
            else:
                s += '%s = %s\n' % (k,v)
    return s


_indent = 0  # number of spaces to indent in __str__ formatting
             # incremented by 2 on each level


def returnNone(key):
    """Always returns None."""
    return None

def failedLookup(key):
    """Raise a KeyError."""
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


    def __init__(self,data={},default=returnNone):
        """Create a new Dict instance.

        The Dict can be initialized with a Python dict or a Dict.
        If defined, default is a function that is used for alternate key
        lookup if the key was not found in the dict.
        """
        dict.__init__(self,data.items())
        self._default_ = default


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
            if self._default_:
                return self._default_(key)
            else:
                raise KeyError


    def __delitem__(self,key):
        """Allow items to be deleted using del self[key].

        Silently ignore if key is nonexistant.
        """
        try:
            dict.__delitem__(self, key)
        except KeyError:
            pass


    def update(self,data={}):
        """Add a dictionary to the Dict object.

        The data can be a dict or Dict type object. 
        """
        dict.update(self,data)


    def __deepcopy__(self,memo):
        """Create a deep copy of ourself."""
        newdict = Dict(default=self.default)
        for k,v in self.items():
            newdict[k] = copy.deepcopy(v,memo)
        return newdict



class Config(Dict):
    """A configuration class allowing Python expressions in the input.

    The configuration settings are stored in the __dict__ of a Python object.
    An item 'foo' in the configuration 'config' can be accessed either as
    dictionary lookup (config['foo']) or as object attribute (config.foo).

    The configuration object can be initialized from a multiline string or
    a text file (or any other object that allows iterating over strings). 
    
    The format of the config file/text is as follows.

    All config lines should have the format: key = value,
    where key is a string and value is a Python expression
    The first '=' character on the line is the delimiter between key and value.
    Blanks around both the key and the value are stripped.
    The value is then evaluated as a Python expression and stored in a variable
    with name specified by the key. This variable is available for use in
    subsequent configuration lines. It is an error to use a variable before
    it is defined.
    The key,value pair is also stored in the config dictionary, unless the key
    starts with an underscore ('_'): this provides for local variables.

    Lines starting with '#' are comments and are ignored, as are empty and
    blank lines.
    A line starting with '[' starts a new section. A section is nothing more
    than a Python dictionary inside the config dictionary. The section name
    is delimited by '['and ']'. All subsequent lines will be stored in the
    section dictionary instead of the toplevel dictionary.

    All other lines are executed as python statements. This allows e.g. for
    importing modules.

    Whole dictionaries can be inserted at once in the config with the
    update() function. 
    
    All defined variables while reading config files remain available for use
    in the config file statements, even over multiple calls to the read()
    function. Variables inserted with addSection() will not be available
    as individual variables though, but can be access as self['name'].
    
    As an example, if your config file looks like
      aa = 'bb'
      bb = aa
      [cc]
      aa = 'aa'
      _n = 3
      rng = range(_n)
    the resulting configuration dictionary is
    {'aa': 'bb', 'bb': 'bb', 'cc': {'aa': 'aa', 'rng': [0, 1, 2]}}

    As far as the resulting Config contents is concerned, the following are
    equivalent:
    C.update({'key':'value'})
    C.read("key='value'\n")
    There is an important difference though: the second line will make a
    variable key (with value 'value') available in subsequent Config read()
    function calls.
    """


    def __init__(self,data={},default=returnNone):
        """Creates a new Config instance.

        The configuration can be initialized with a dictionary, or
        with a variable that can be passed to the read() function.
        The latter includes the name of a config file, or a multiline string
        holding the contents of a configuration file.
        """
        Dict.__init__(self,default=default)
        if isinstance(data,dict):
            self.update(data)
        elif data:
            self.read(data)


    def update(self,data={},name=None,removeLocals=False):
        """Add a dictionary to the Config object.

        The data, if specified, should be a valid Python dict.
        If no name is specified, the data are added to the top dictionary
        and will become attributes.
        If a name is specified, the data are added to the named attribute,
        which should be a dictionary. If the name does not specify a
        dictionary, an empty one is created, deleting the existing attribute.

        If a name is specified, but no data, the effect is to add a new
        empty dictionary (section) with that name.

        If removeLocals is set, keys starting with '_' are removed from the
        data before updating the dictionary and not
        included in the config. This behaviour can be changed by setting
        removeLocals to false.
        """
        if removeLocals:
            for k in data.keys():
                if k[0] == '_':
                    del data[k]
        if name:
            if not self.has_key(name) or not isinstance(self[name],dict):
                self[name] = Dict()
            self[name].update(data)
        else:
            Dict.update(self,data)

    
    def _read_error(self,filename,lineno,line):
        if filename:
            where = 'config file %s,' % filename
        else:
            where = ''
        raise RuntimeError,'Error in %s line %d:\n%s' % (where,lineno,line)


    def read(self,fil,debug=False):
        """Read a configuration from a file or text

        fil is a sequence of strings. Any type that allows a loop like 
          for line in fil:
        to iterate over its text lines will do. This could be a file type, or
        a multiline text after splitting on '\n'.

        The function will try to react intelligently if a string is passed as
        argument. If the string contains at least one '\n', it will be
        interpreted as a multiline string and be splitted on '\n'.
        Else, the string will be considered and a file with that name will
        be opened. It is an error if the file does not exist or can not be
        opened.
        
        The function returns self, so that you can write: cfg = Config().
        
        """
        filename = None
        if type(fil) == str:
            if fil.find('\n') >= 0:
                fil = fil.split('\n')
            else:
                filename = fil
                fil = file(fil,'r')
        section = None
        contents = {}
        lineno = 0
        for line in fil:
            lineno += 1
            s = line.strip()
            if len(s)==0 or s[0] == '#':
                continue
            elif s[0] == '[':
                if contents:
                    self.update(name=section,data=contents,removeLocals=True)
                    contents = {}
                i = s.find(']')
                if i<0:
                    self.read_error(filename,lineno,line)
                section = s[1:i]
                if debug:
                    print "Starting new section '%s'" % section
                continue
            else:
                if debug:
                    print "READ: "+line
            i = s.find('=')
            if i >= 0:
                key = s[:i].strip()
                if len(key) == 0:
                    self.read_error(filename,lineno,line)
                contents[key] = eval(s[i+1:].strip())
                globals().update(contents)
            else:
                exec(s)
        if contents:
            self.update(name=section,data=contents,removeLocals=True)
        return self


    def __getitem__(self, key):
        """Allows items to be addressed as self[key].

        This is equivalent to the Dict lookup, except that items in
        subsections can also be retrieved with a single key of the format
        section/key.
        While this lookup mechanism works for nested subsections, the syntax
        for config files allows for only one level of sections!
        Also beware that because of this functions, no '/' should be used
        inside normal keys and sections names.
        """
        i = key.rfind('/')
        if i == -1:
            return Dict.__getitem__(self, key)
        else:
            res = self[key[:i]][key[i+1:]]
            if res:
                return res
            else:
                return self._default_(key)



    def __setitem__(self, key, val):
        """Allows items to be set as self[section/key] = val.

        """
        i = key.rfind('/')
        if i == -1:
            self.update({key:val})
        else:
            self.update({key[i+1:]:val},key[:i])


    def __str__(self):
        """Format the Config in a way that can be read back."""
        s = "# Config written by pyFormex\n\n"
        for k,v in self.iteritems():
            if not isinstance(v,Dict):
                s += dicttostr({k:v})
        for k,v in self.iteritems():
            if isinstance(v,Dict):
                s += "\n[%s]\n" % k
                s += dicttostr(v)
        s += "\n# End of config\n"
        return s


if __name__ == '__main__':


    C = Config("""# A simple config example
aa = 'bb'
bb = aa
[cc]
aa = 'aa'    # yes ! comments are allowed (they are stripped by eval())
_n = 3       # local: will get stripped
rng = range(_n)
""")
    print C
    print C['aa']
    print C['cc']
    print C['cc/aa']
    print C['dd']


    def reflookup(key):
        return C[key]

    D = Config(default = reflookup)

    print D
    print D['aa']
    print D['cc']
    print D['cc/aa']
    print D['dd']

    D['aa'] = 'wel'
    D['dd'] = 'hoe'
    D['cc/aa'] = 'ziedewel'
    print D
    print C
    print D['cc/aa']
    print D['cc/rng']
    
