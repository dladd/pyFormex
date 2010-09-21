#!/usr/bin/env python
# $Id$
#
"""create_manifest.py

This script creates the MANIFEST file which lists all the files
to be included in a pyFormex source distribution.
"""
from pyformex.utils import listTree

files = [
    'README',
    'post-install',
    'pyformex-pyformex.desktop',
    'pyformex-viewer',
    'setup.py',
    ] + \
    listTree('pyformex',listdirs=False,sorted=True,
             excludedirs=['.svn'],
             includedirs=['gui','plugins'],
             includefiles=['.*\.py$','pyformex(rc)?$']) + \
    listTree('pyformex/icons',listdirs=False,sorted=True,
             excludedirs=['.svn'],
             includefiles=['.*\.xpm$','pyformex.*\.png$']) + \
    listTree('pyformex/lib',listdirs=False,sorted=True,
             excludedirs=['.svn'],
             includefiles=['.*\.c$','configure(_py)?$','Makefile.in$','__init__.py$']) + \
    listTree('pyformex/examples',listdirs=False,sorted=True,
             excludedirs=['.svn'],
             excludefiles=['.*\.pyc','.*~$'],
             includefiles=['.*\.py$','scripts.cat','README']
             ) + \
    listTree('pyformex/data',listdirs=False,sorted=True,
             excludedirs=['.svn'],
             excludefiles=['.*\.pyc','.*~$']
             ) + \
    listTree('pyformex/doc',listdirs=False,sorted=True,
             excludedirs=['.svn'],
             includefiles=['COPYING$','README$','ReleaseNotes$']
             ) + \
    listTree('pyformex/doc/html',listdirs=False,sorted=True,
             excludedirs=['.svn'],
             ) + \
    listTree('pyformex/external',listdirs=False,sorted=True,
             excludedirs=['.svn'],
             excludefiles=['.*~$'],
             )

for f in files:
   print 'include %s' % f

# End

