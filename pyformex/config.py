#!/usr/bin/env python
##
##  This file is part of pyFormex 0.8.2 Release Sat Jun  5 10:49:53 2010
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
"""A general yet simple configuration class.

| (C) 2005 Benedict Verhegghe
| Distributed under the GNU GPL version 3 or later

Why
   I wrote this simple class because I wanted to use Python
   expressions in my configuration files. This is so much more fun
   than using .INI style config files.  While there are some other
   Python config modules available on the web, I couldn't find one
   that suited my needs and my taste: either they are intended for
   more complex configuration needs than mine, or they do not work
   with the simple Python syntax I expected.

What
   Our Config class is just a normal Python dictionary which can hold
   anything.  Fields can be accessed either as dictionary lookup
   (config['foo']) or as object attributes (config.foo).  The class
   provides a function for reading the dictionary from a flat text
   (multiline string or file). I will always use the word 'file'
   hereafter, because that is what you usually will read the
   configuration from.  Your configuration file can have named
   sections. Sections are stored as other Python dicts inside the top
   Config dictionary. The current version is limited to one level of
   sectioning.
"""

import copy
from mydict import Dict,returnNone


def formatDict(d):
    """Format a dict in Python source representation.

    Each (key,value) pair is formatted on a line of the form::
    
       key = value
       
    The resulting text is a legal Python script to define the items in the
    dict.
    """
    s = ""
    if isinstance(d,dict):
        for k,v in d.iteritems():
            if type(v) == str:
                s += '%s = "%s"\n' % (k,v)
            else:
                s += '%s = %s\n' % (k,v)
    return s


class Config(Dict):
    """A configuration class allowing Python expressions in the input.

    The configuration settings are stored in the __dict__ of a Python
    object.  An item 'foo' in the configuration 'config' can be accessed
    either as dictionary lookup (``config['foo']``) or as object attribute
    (``config.foo``).

    The configuration object can be initialized from a multiline string or
    a text file (or any other object that allows iterating over strings).
    
    The format of the config file/text is described hereafter.
   
    All config lines should have the format: key = value, where key is a
    string and value is a Python expression The first '=' character on the
    line is the delimiter between key and value.  Blanks around both the
    key and the value are stripped.  The value is then evaluated as a
    Python expression and stored in a variable with name specified by the
    key. This variable is available for use in subsequent configuration
    lines. It is an error to use a variable before it is defined.  The
    key,value pair is also stored in the config dictionary, unless the key
    starts with an underscore ('_'): this provides for local variables.

    Lines starting with '#' are comments and are ignored, as are empty
    and blank lines.  Lines ending with '\' are continued on the next
    line.  A line starting with '[' starts a new section. A section is
    nothing more than a Python dictionary inside the config
    dictionary. The section name is delimited by '['and ']'. All
    subsequent lines will be stored in the section dictionary instead
    of the toplevel dictionary.

    All other lines are executed as python statements. This allows
    e.g. for importing modules.

    Whole dictionaries can be inserted at once in the config with the
    update() function.
    
    All defined variables while reading config files remain available
    for use in the config file statements, even over multiple calls to
    the read() function. Variables inserted with addSection() will not
    be available as individual variables though, but can be access as
    ``self['name']``.
    
    As an example, if your config file looks like::
    
       aa = 'bb'
       bb = aa
       [cc]
       aa = 'aa'
       _n = 3
       rng = range(_n)
       
    the resulting configuration dictionary is
    ``{'aa': 'bb', 'bb': 'bb', 'cc': {'aa': 'aa', 'rng': [0, 1, 2]}}``

    As far as the resulting Config contents is concerned, the following are
    equivalent::
    
       C.update({'key':'value'})
       C.read("key='value'\\n")

    There is an important difference though: the second line will make a
    variable key (with value 'value') available in subsequent Config read()
    method calls.   
    """

    def __init__(self,data={},default=None):
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

        `fil` is a sequence of strings. Any type that allows a loop like 
        ``for line in fil:``
        to iterate over its text lines will do. This could be a file type, or
        a multiline text after splitting on '\\n'.

        The function will try to react intelligently if a string is passed as
        argument. If the string contains at least one '\\n', it will be
        interpreted as a multiline string and be splitted on '\\n'.
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
        continuation = False
        for line in fil:
            lineno += 1
            ls = line.strip()
            if len(ls)==0 or ls[0] == '#':
                continue
            if continuation:
                s += ls
            else:
                s = ls
            continuation = s[-1] == '\\'
            if continuation:
                s = s[:-1]
                continue
            if s[0] == '[':
                if contents:
                    self.update(name=section,data=contents,removeLocals=True)
                    contents = {}
                i = s.find(']')
                if i<0:
                    self.read_error(filename,lineno,line)
                section = s[1:i]
                if debug:
                    print("Starting new section '%s'" % section)
                continue
            else:
                if debug:
                    print("READ: "+line)
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


    def __setitem__(self, key, val):
        """Allows items to be set as self[section/key] = val.

        """
        i = key.rfind('/')
        if i == -1:
            self.update({key:val})
        else:
            self.update({key[i+1:]:val},key[:i])


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
            try:
                return self[key[:i]][key[i+1:]]
            except KeyError:
                return self._default_(key)


    def __delitem__(self,key):
        """Allows items to be delete with del self[section/key].

        """
        i = key.rfind('/')
        if i == -1:
            Dict.__delitem__(self,key)
        else:
            del self[key[:i]][key[i+1:]]


    def __str__(self):
        """Format the Config in a way that can be read back.

        This function is mostly used to format the data for writing it to
        a configuration file. See the write() method.

        The return value is a multiline string with Python statements that can
        be read back through Python to recreate the Config data. Usually
        this is done with the Config.read() method.
        """
        s = ''
        for k,v in self.iteritems():
            if not isinstance(v,Dict):
                s += formatDict({k:v})
        for k,v in self.iteritems():
            if isinstance(v,Dict):
                s += "\n[%s]\n" % k
                s += formatDict(v)
        return s


    def write(self,filename,header="# Config written by pyFormex    -*- PYTHON -*-\n\n",trailer="\n# End of config\n"):
        """Write the config to the given file

        The configuration data will be written to the file with the given name
        in a text format that is both readable by humans and by the
        Config.read() method.
        
        The header and trailer arguments are strings that will be added at
        the start and end of the outputfile. Make sure they are valid
        Python statements (or comments) and that they contain the needed
        line separators, if you want to be able to read it back.
        """
        fil = file(filename,'w')
        fil.write(header)
        fil.write("%s" % self)
        fil.write(trailer)
        fil.close()
        

    def keys(self,descend=True):
        """Return the keys in the config.

        By default this descends one level of Dicts.
        """
        keys = Dict.keys(self)
        if descend:
            for k,v in self.iteritems():
                if isinstance(v,Dict):
                    keys += ['%s/%s' % (k,ki) for ki in v.keys()]
                
        return keys


if __name__ == '__main__':


    def show(s):
        try:
            v = eval(s)
            print("%s = %s" % (s,v))
        except:
            print("%s ! ERROR" % s)
        
    C = Config("""# A simple config example
aa = 'bb'
bb = aa
[cc]
aa = 'aa'    # yes ! comments are allowed (they are stripped by eval())
_n = 3       # local: will get stripped
rng = range(_n)
""")
    show("C")
    show("C['aa']")
    show("C['cc']")
    show("C['cc/aa']")
    show("C['dd']")


    def reflookup(key):
        return C[key]

    D = Config(default = reflookup)

    show("D")
    show("D['aa']")
    show("D['cc']")
    show("D['cc/aa']")
    show("D['dd']")

    D['aa'] = 'wel'
    D['dd'] = 'hoe'
    D['cc/aa'] = 'ziedewel'
    show("D")
    show("C")
    show("D['cc/aa']")
    show("D['cc/rng']")
    print("BUT!!!!")
    show("D['cc']")

    # This should give an error
    show("D['ee']")
    show("D.get('ee','NO Error')")
    show("D.get('cc/ee','NO Error')")

    D['cc/bb'] = 'ok'
    show("D.keys()")
    del D['aa']
    del D['cc/aa']
    show("D.keys()")
    del D['cc']
    show("D.keys()")


# End
   
    
    
    
    
    
