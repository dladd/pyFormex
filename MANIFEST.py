#!/usr/bin/env python
# $Id$
##
##  This file is part of the pyFormex project.
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  https://savannah.nongnu.org/projects/pyformex/
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
"""MANIFEST.py

This script creates the MANIFEST file which lists all the files
to be included in a pyFormex source distribution.
"""
from pyformex.utils import listTree

files = [
    'README',
    'post-install',
    'pyformex-pyformex.desktop',
    'pyformex-viewer',
    'pyformex-search',
    'setup.py',
    ] + \
    listTree('pyformex',listdirs=False,sorted=True,
             excludedirs=['.svn'],
             includedirs=['gui','plugins'],
             includefiles=['.*\.py$','pyformex(rc)?$']
             ) + \
    listTree('pyformex/icons',listdirs=False,sorted=True,
             excludedirs=['.svn'],
             includefiles=['README','.*\.xpm$','pyformex.*\.png$']
             ) + \
    listTree('pyformex/lib',listdirs=False,sorted=True,
             excludedirs=['.svn'],
             includefiles=['.*\.c$','.*\.py$','configure(_py)?$','Makefile.in$']
             ) + \
    listTree('pyformex/examples',listdirs=False,sorted=True,
             excludedirs=['.svn'],
             excludefiles=['.*\.pyc','.*~$'],
             includefiles=['.*\.py$','scripts.cat','README']
             ) + \
    listTree('pyformex/data',listdirs=False,sorted=True,
             excludedirs=['.svn','benchmark'],
             excludefiles=['.*\.pyc','.*~$','PTAPE.*'],
             includefiles=[
                'README',
                'benedict_6.jpg',
                'blippo.pgf',
                'butterfly.png',
                'hesperia-nieve.prop',
                'horse.off',
                'horse.pgf',
                'materials.db',
                'sections.db',
                'splines.pgf',
                'supershape.txt',
                'teapot.off',
                'world.jpg',
                ],
             ) + \
    listTree('pyformex/doc',listdirs=False,sorted=True,
             excludedirs=['.svn'],
             includefiles=['COPYING$','README$','ReleaseNotes$']
             ) + \
    listTree('pyformex/doc/html',listdirs=False,sorted=True,
             excludedirs=['.svn'],
             ) + \
    listTree('pyformex/external',listdirs=False,sorted=True,
             excludedirs=['.svn','pyftgl','sippy-ftgl'],
             excludefiles=['.*~$'],
             ) + \
    listTree('pyformex/bin',listdirs=False,sorted=True,
             excludedirs=['.svn'],
             excludefiles=['.*~$'],
             )

for f in files:
   print 'include %s' % f

# End

