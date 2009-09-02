#!/usr/bin/env python
"""Create a basic Sphinx source file for a Python module.

Sphinx is the next generation documentation system for Python.
It's input is written in ReST (ReStructuredText)
This script creates a basic ReST source to create a reference manual for
the classes and functions defined in a Python module.

Usage:  py2sphinx.py PYTHONFILE [> outputfile.tex]

(C) 2009 Benedict Verhegghe (benedict.verhegghe@ugent.be)
I wrote this software in my free time, for my joy, not as a commissioned task.
Any copyright claims made by my employer should therefore be considered void.
"""

import os,sys

############# Output formatting ##########################

def find_doc(fil):
    """Reads from a text file until a docstring is found.

    Returns the docstring.
    """
    s = ""
    for line in fil:
        i = line.find('"""')
        if i >= 0:
            if s:
                s += line[:i]
                break
            else:
                s = line[i+3:]
        elif s:
            s += line
    return s


def split_doc(docstring):
    s = docstring.split('\n')
    shortdoc = s[0]
    if len(s) > 2:
        longdoc = '\n'.join(s[2:])
    else:
        longdoc = ''
    return shortdoc.strip('"'),longdoc.strip('"')
    

def underline(s,c='='):
    """Return a string underlined with character c"""
    return s + '\n' + (c * len(s)) + '\n'
    

def ship_module(fil,name,shortdoc,longdoc):
    fil.write (""".. $%s$  -*- rst -*-
.. pyformex reference manual --- %s
.. CREATED WITH quickref.py: DO NOT EDIT

.. include:: defines.inc
.. include:: links.inc

.. _sec:ref-%s:

%s

.. automodule:: %s
   :synopsis: %s
   :members:
   
.. moduleauthor:: 'pyFormex project' <'http://pyformex.berlios.de'>

.. End
""" % ('Id',name,name,underline(":mod:`%s` --- %s" % (name,shortdoc)),name,shortdoc))


def main(args):
    sys.path[0:0] = [os.path.abspath('../pyformex')]
    for name in args:
        s = name.split('.')
        if len(s) == 2:
            pkg,mdl = s
        else:
            pkg,mdl = None,name
        module = __import__(name)
        if pkg is not None:
            module = getattr(module,mdl)
        doc = module.__doc__
        shortdoc,longdoc = split_doc(doc)
        outfile = "ref-%s.rst" % mdl
        print "%s -> %s" % (name,outfile)
        fil = file(outfile,'w')
        ship_module(fil,name,shortdoc,longdoc)


if __name__ == "__main__":
    main(sys.argv[1:])

# End
