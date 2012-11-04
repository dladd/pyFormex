#!/usr/bin/env python
# $Id$
##
##  This file is part of pyFormex 0.8.8  (Sun Nov  4 15:24:17 CET 2012)
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
"""Extract info from a Python file and shipout in TeX format.

This script automatically extracts class & function docstrings and argument
list from a module and ships out the information in LaTeX format.

(C) 2009 Benedict Verhegghe (benedict.verhegghe@ugent.be)
I wrote this software in my free time, for my joy, not as a commissioned task.
Any copyright claims made by my employer should therefore be considered void.

It includes parts from the examples in the Python library reference manual
in the section on using the parser module. Refer to the manual for a thorough
discussion of the operation of this code.

Usage:  py2tex.py PYTHONFILE [> outputfile.tex]
"""

import os,sys

# set path to the pyformex modules
parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pyformexdir = os.path.join(parent,'pyformex')
for d in [ 'lib', 'gui', 'plugins' ]: # this order is important!
    sys.path.insert(0,os.path.join(pyformexdir,d))
sys.path.insert(0,pyformexdir)
sys.path.insert(0,os.path.dirname(pyformexdir))

from pyformex.odict import ODict


import inspect


def filter_names(info):
    return [ i for i in info if not i[0].startswith('_') ]


def filter_docstrings(info):
    return [ i for i in info if not (i[1].__doc__ is None or i[1].__doc__.startswith('_')) ]

def filter_module(info,modname):
    return [ i for i in info if i[1].__module__ == modname]

def function_key(i):
    return i[1].func_code.co_firstlineno

def class_key(i):
    methods = inspect.getmembers(i[1],inspect.ismethod)
    return inspect.getsourcelines(i[1])[1]


def do_class(name,obj):
    # get class methods #
    methods = inspect.getmembers(obj,inspect.ismethod)
    methods = filter_names(methods)
    methods = filter_docstrings(methods)
    methods = sorted(methods,key=function_key)
    names = [ f[0] for f in methods ]
    ship_class(name,names)


def do_module(filename):
    """Retrieve information from the parse tree of a source file.

    filename
        Name of the file to read Python source code from.
    """
    modname = inspect.getmodulename(filename)
    #print "MODULE %s" % modname
    #print inspect.getmoduleinfo(filename)
    module = __import__(modname)
    classes = [ c for c in inspect.getmembers(module,inspect.isclass) if c[1].__module__ == modname ]
    classes = filter_names(classes)
    classes = filter_docstrings(classes)
    #print classes
    #print [ class_key(i) for i in classes ]
    classes = sorted(classes,key=class_key)
    #print classes

    # Functions #
    functions = [ c for c in inspect.getmembers(module,inspect.isfunction) if c[1].__module__ == modname ]
    functions = filter_names(functions)
    functions = filter_docstrings(functions)
    functions = sorted(functions,key=function_key)
    #print "FUNCTIONS"
    names = [ f[0] for f in functions ]
    #print names

    # Shipout
    
    ship_start()
    ship_module(modname,module.__doc__)
    #print names
    ship_functions(names)
    ship_class_init(modname)
    for c in classes:
        do_class(*c)
    ship_functions_init(modname)
    ship_end()
    
    sys.__stdout__.write(out)

       

############# Output formatting ##########################


def split_doc(docstring):
    s = docstring.split('\n')
    shortdoc = s[0]
    if len(s) > 2:
        longdoc = '\n'.join(s[2:])
    else:
        longdoc = ''
    return shortdoc.strip('"'),longdoc.strip('"')


def sanitize(s):
    """Sanitize a string for LaTeX."""
    for c in '#&%':
        s = s.replace('\\'+c,c)
        s = s.replace(c,'\\'+c)
    ## for c in '{}':
    ##     s = s.replace('\\'+c,c)
    ##     s = s.replace(c,'\\'+c)
    return s

    

out = ''

def ship_start():
    global out
    out = ''

def ship(s):
    global out
    out += s
    out += '\n'
    

def debug(s):
    if options.debug:
        ship('.. DEBUG:'+str(s))

def ship_module(name,docstring):
    shortdoc,longdoc = split_doc(docstring)
    ship(""".. $%s$  -*- rst -*-
.. pyformex reference manual --- %s
.. CREATED WITH py2rst.py: DO NOT EDIT

.. include:: <isonum.txt>
.. include:: ../defines.inc
.. include:: ../links.inc

.. _sec:ref-%s:

:mod:`%s` --- %s
%s

.. automodule:: %s
   :synopsis: %s""" % ('Id',name,name,name,shortdoc,'='*(12+len(name)+len(shortdoc)),name,shortdoc))

def ship_end():
    ship("""
   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End
""")

def ship_functions(members=[]):
    if members:
        ship("""   :members: %s""" % (','.join(members)))

def ship_class(name,members=[]):
    ship("""
   .. autoclass:: %s
      :members: %s""" % (name,','.join(members)))

def ship_class_init(name):
    ship("""
   ``Classes defined in module %s``
""" % name)

def ship_functions_init(name):
    ship("""
   ``Functions defined in module %s`` 
""" % name)


def main(argv):
    global options,source

    sys.stdout = sys.stderr
    
    from optparse import OptionParser,make_option
    parser = OptionParser(
        usage = """usage: %prog [Options] PYTHONFILE
Creates a reference manual in sphinx format for the functions and classes
defined in PYTHONFILE.""",
        version = "%s (C) 2009 Benedict Verhegghe" % __file__,
        option_list=[
        make_option('-d',"--debug", help="print debug info",
                    action="store_true", dest="debug", default=False),
        make_option('-c',"--continue", help="continue on errors",
                    action="store_false", dest="error", default=True),
        ])
    options, args = parser.parse_args(argv)
    
    for source in args:
        do_module(source)


if __name__ == "__main__":

    import sys
    
    main(sys.argv[1:])

# End
