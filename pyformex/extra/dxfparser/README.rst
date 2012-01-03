.. $Id$
  
..
  This file is part of pyFormex 0.8.5  (Sun Dec  4 21:24:46 CET 2011)
  pyFormex is a tool for generating, manipulating and transforming 3D
  geometrical models by sequences of mathematical operations.
  Home page: http://pyformex.org
  Project page:  http://savannah.nongnu.org/projects/pyformex/
  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be)
  Distributed under the GNU General Public License version 3 or later.
  
  
  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see http://www.gnu.org/licenses/.
  
  

dxfparser
---------

dxfparser converts an AutoCAD DXF file to an ascii format, allowing easy
inspection and further processing by other tools.

dxfparser relies on DXFlib (http://www.qcad.org/dxflib.html) for
parsing the DXF files. This exporter to ascii format was written by
Benedict Verhegghe with the purpose of importing some DXF files into
pyFormex (http://pyformex.org).

Copyright (C) 2011 Benedict Verhegghe <benedict.verhegghe@ugent.be>

This software is distributed under the GPL v3 or higher. 

install
-------

To compile dxfparser, you need to first install the DXFlib library and
development files. On Debian GNU/Linux (or Ubuntu), you can just
install the package 'libdxflib-dev'::

  apt-get install libdxflib

Then, inside this directory, run::

  make
  make install (with root privileges)

This will install the executable program 'dxfparser' in
/usr/local/bin. Change the path in Makefile if you want to install elsewhere.

usage
-----
::

  dxfparser DXFfile(s)

will write the (recognized) contents of the named DXF files to
standard output. It will write the name of processed files to standard
error.
Thus you can use the syntax::

  dxfparser DXFfile.dxf > DXFfile.txt

to get an ascii file with the contents.

The ascii output looks like:


 
.. End
