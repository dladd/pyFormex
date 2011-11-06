#!/usr/bin/env python
# $Id$
##
##  This file is part of pyFormex
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
#
"""stamp-unversioned.py

This script inserts the pyFormex copyright and license statement into the
files with static information and the files that are not included into
the release tarballs.

This should only be run when some information in the Stamp.template
or the list of files changes.
"""
from pyformex.utils import listTree

files = [
    __file__,
    'Description',
    'History',
    'HOWTO-dev.rst',
    'MANIFEST.py',
    'add_Id',
    'create_revision_graph',
    'install-pyformex-svn-desktop-link',
    'pyformex-viewer',
    'searchpy',
    'sloc.py',
    ] #+ \
    ## listTree('pyformex',listdirs=False,sorted=True,
    ##          excludedirs=['.svn'],
    ##          includedirs=['gui','plugins'],
    ##          includefiles=['.*\.py$','pyformex(rc)?$']
    ##          ) + \
    ## listTree('pyformex/icons',listdirs=False,sorted=True,
    ##          excludedirs=['.svn'],
    ##          includefiles=['README','.*\.xpm$','pyformex.*\.png$']
    ##          ) + \
    ## listTree('pyformex/lib',listdirs=False,sorted=True,
    ##          excludedirs=['.svn'],
    ##          includefiles=['.*\.c$','.*\.py$','configure(_py)?$','Makefile.in$']
    ##          ) + \
    ## listTree('pyformex/examples',listdirs=False,sorted=True,
    ##          excludedirs=['.svn'],
    ##          excludefiles=['.*\.pyc','.*~$'],
    ##          includefiles=['.*\.py$','scripts.cat','README']
    ##          ) + \
    ## listTree('pyformex/data',listdirs=False,sorted=True,
    ##          excludedirs=['.svn'],
    ##          excludefiles=['.*\.pyc','.*~$','PTAPE.*'],
    ##          ) + \
    ## listTree('pyformex/doc',listdirs=False,sorted=True,
    ##          excludedirs=['.svn'],
    ##          includefiles=['COPYING$','README$','ReleaseNotes$']
    ##          ) + \
    ## listTree('pyformex/doc/html',listdirs=False,sorted=True,
    ##          excludedirs=['.svn'],
    ##          ) + \
    ## listTree('pyformex/external',listdirs=False,sorted=True,
    ##          excludedirs=['.svn','pyftgl','sippy-ftgl'],
    ##          excludefiles=['.*~$'],
    ##          ) + \
    ## listTree('pyformex/bin',listdirs=False,sorted=True,
    ##          excludedirs=['.svn'],
    ##          excludefiles=['.*~$'],
    ##          )

print files


# End

