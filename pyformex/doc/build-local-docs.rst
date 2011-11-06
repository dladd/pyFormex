.. $Id$
  
..
  This file is part of pyFormex 0.8.5     Sun Nov  6 17:27:05 CET 2011
  pyFormex is a tool for generating, manipulating and transforming 3D
  geometrical models by sequences of mathematical operations.
  Home page: http://pyformex.org
  Project page:  https://savannah.nongnu.org/projects/pyformex/
  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be)
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
  
  

How to build pyFormex documentation
===================================

1. Install Sphinx
-----------------
The pyFormex documentation is built using Sphinx. The location of the
source files and the build environment is the `sphinx` directory under
the `trunk`.

In order to build the documentation, you need to have Sphinx installed.
The versions of Sphinx available from debian or ubuntu repositories are
often not recent enough and need to be patched. Recent Sphinx packages can
be obtained from PyPi, but its easy-install installation procedure does
not work very well with debian/ubuntu (and is strongly discouraged).

The best thing to do is install Sphinx using stdeb::

   apt-get install stdeb python-jinja2
   apt-file update
   pypi-install Sphinx
   

2. Patch Sphinx
---------------
Sphinx needs a small patch to build the pyFormex reference manual.
The patch file (for sphinx 0.6.6 and 1.0.4) is found in the pyFormex 
`sphinx` directory. Pathcing has to be done as root, so be careful.

First find out where Sphinx was installed::

  python -c "import sphinx; print sphinx.__path__"

This will print the path to your sphinx installation, something like::

  ['/usr/lib/pymodules/python2.6/sphinx']

Go to the path where spinx is installed::

  cd /usr/lib/pymodules/python2.6/sphinx

Then perform the following command as root. Substitute your full path to
your pyformex svn sources and use the correct version in the .diff file 
(1.0.4 also works for 1.0.5). This is just a dry run, so do not be afraid 
to break something::

  patch -p 1 --dry-run < PATH-TO-PYFORMEX-SVN/sphinx/sphinx-1.0.4-bv.diff 

If the patch command above succeeds in trying to apply the patches 
(you get messages like 'Hunk #? succeeded at ??'), finally apply
the patch by repeating the same command without the --dry-run option::

  patch -p 1 < PATH-TO-PYFORMEX-SVN/sphinx/sphinx-1.0.4-bv.diff 


3. Build the documentation
--------------------------
You can now build the documentation by executing the command::

  make html

either in the `sphinx` directory, or in the level above it.
If the build succeeded (a few warnings may be emitted), install it in the
`doc/html` directory of your pyformex svn tree with::

  make svndoc

The new documentation will now be accessible when running pyformex from the
svn sources.

4. For developers
-----------------
Do not execute the ```make svndoc``` command if the ```make html``` produced
many errors/warnings. Once installed under `doc`, the faulty documentation
files would be inserted in the repositories when you check in, 
and thus be distributed to others.

.. End




