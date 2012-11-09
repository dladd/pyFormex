# $Id$
##
##  This file is part of pyFormex 0.8.9  (Fri Nov  9 10:49:51 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2012 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
Interface with Calculix FE input files (.inp).

"""

import re

## re_command = re.compile("*([*A-Za-z]+)[ ,]*(.*)")

## def readCommand(line):
##     """Read a command line, return the command and a dict with options"""
##     m = re_command.match(line)
##     if not m:
##         raise ValueError,"Invalid/unrecognized command line:\n%s" % line

##     cmd = m.groups()[0]
##     opts = eval('dict(m.groups()[1]


def readCommand(line):
    """Read a command line, return the command and a dict with options"""
    if line[0] == '*':
        line = line[1:]
    s = line.split(',')
    s = [si.strip() for si in s]
    cmd = s[0]
    opts = {}
    for si in s[1:]:
        kv = si.split('=')
        k = kv[0]
        if len(kv) > 1:
            v = kv[1]
        else:
            v = True
        opts[k] = v
    return cmd,opts


def readInput(fn):
    """Read an input file (.inp)"""
    with open(fn) as fil:
        line = fil.next()
        while line:
            if line.startswith('*'):
                if line[1] != '*':
                    cmd,opts = readCommand(line[1:])
                    print("Keyword %s; Options %s" % (cmd,opts))
            line = fil.next()
