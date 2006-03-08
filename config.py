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

from mydict import Dict

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
    the read configuration dictionary be
    {'aa': 'bb', 'bb': 'bb', 'cc': {'aa': 'aa', 'rng': [0, 1, 2]}}

    As far as the resulting Config contents is concerned, the following are
    equivalent:
    C.update({'key':'value'})
    C.read("key='value'\n")
    There is an important difference though: the second line will make a
    variable key (with value 'value') available in subsequent Config read()
    function calls.
    """


    def __init__(self,data={}):
        """Creates a new Config instance.

        The configuration can be initialized with a dictionary, or
        with a variable that can be passed to the read() function.
        The latter includes the name of a config file, or a multiline string
        holding the contents of a configuration file.
        """
        Dict.__init__(self)
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
            if not self[name] or not isinstance(self[name],Dict):
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


    def __str__(self):
        """Format the Config in a way that can be red back."""
        s = "# Config written by pyFormex\n\n"
        for k,v in self.iteritems():
            if not isinstance(v,Dict):
                s += dicttostr({k:v})
        for k,v in self.iteritems():
            if isinstance(v,Dict):
                s += "\n[%s]\n" % k
                s += dicttostr(v)
        return s
    

if __name__ == '__main__':

    C = Config("""# A simple config example
aa = 'bb'
bb = aa
[cc]
aa = 'aa'    # yes ! comments are allowed (they are stripped by eval())
_n = 3
rng = range(_n)
""")
    print C
    print C.aa
    print C['aa']
    print C.get('dd',1)  # 1
    print C['dd']        # None
    print C.dd           # None
    # beware for this though!
    print C.get
    print C['get']
    C['get'] = 'hallo'
    print C
    print C.get('aa',1)
    print C.get('ab',1)
        
