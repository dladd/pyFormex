# $Id$
##
##  This file is part of pyFormex 0.8.5  (Sun Dec  4 21:24:46 CET 2011)
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

from distutils.core import setup, Extension
setup(name="pygl2ps",
      version="1.3.3",
      description="Wrapper for GL2PS, an OpenGL to PostScript Printing Library",
      author="Benedict Verhegghe",
      author_email="benedict.verhegghe@ugent.be",
      url="http://pyformex.org",
      long_description="""
Python wrapper for GL2PS library by Christophe Geuzaine.
See http://www.geuz.org/gl2ps/
""",
from __future__ import print_function
      license="GNU LGPL (Library General Public License)",
      py_modules=["gl2ps"],
      ext_modules=[Extension("_gl2ps",
                             ["gl2ps.c","gl2ps_wrap.c"],
                             libraries=["GL"])])
