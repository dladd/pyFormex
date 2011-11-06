.. $Id$    -*- rst -*-
  
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
  
  

==============================================
Guidelines for creating pyFormex documentation
==============================================

- Documentation is written in ReST (ReStructuredText)

- Start the .rst file with the following header::

  .. $Id$
  .. pyformex documentation --- chaptername
  ..
  .. include:: defines.inc
  .. include:: links.inc
  ..
  .. _cha:partname:


- Replace in this header chaptername with the documentation chapter name.

- See also the following links for more information:

  - guidelines for documenting Python: http://docs.python.org/documenting/index.html
  - Sphinx documentation: http://sphinx.pocoo.org/
  - ReStructuredText page of the docutils project: http://docutils.sourceforge.net/rst.html

- When refering to pyFormex as the name of the software or project,
  always use the upper case 'F'. When referring to the command to run
  the program, or a directory path name, use all lower case: ``pyformex``.


- Install required packages :: 
    apt-get install dvipng
    apt-get install python-sphinx

- Patch the sphinx installation.

  We use a slightly patched version of Sphinx. The pyformex/sphinx
  source tree contains the required patch file sphinx-1.04-bv.diff. It
  was created for Sphinx 1.0.4 but will still work for slightly newer
  versions (tested on 1.0.8). Do the following as root::

    cd /usr/share/pyshared/sphinx
    patch -p1 --dry-run < ???/pyformex/sphinx/sphinx-1.0.4-bv.diff

  This will only test the patching. If all hunks succeed, run the
  command again without the '--dry-run'::

    patch -p1 < ???/pyformex/sphinx/sphinx-1.0.4-bv.diff

- To create the html documentation, do ``make html`` in the ``sphinx`` directory.

- To convert LaTeX source to ReST, you can use the converter from the Python 
  doctools. Best is to install the doctools from Subversion sources, in your
  top level pyformex directory, using the command::

   svn co http://svn.python.org/projects/doctools/converter 

  Do not use the ``convert.py`` script in the created ``coverter`` directory.
  Use the script in the ``sphinx`` directory, as follows::

   ./convert.py latexsource.tex

  This will ``create latexsource.rst``.

  If you have defined new LaTeX commands, the converter may not be able to
  convert them: either change, remove or uncomment them. Uncommenting will
  copy the commands to the output, so that you can fix them afterwards in the
  ``.rst`` file.

Images
======

- Put original images int the subdirectory ``images``.

- Create images with a transparent or white background.

- Use PNG images whenever possible.

- Create the reasonable size for inclusion on a web page. Use a minimal canvas size and maximal zooming.

- Give related images identical size (set canvas size and use autozoom).

- Make composite images to combine multiple small images in a single large one.
  If you have ``ImageMagick``, the following command create a horizontal
  composition ``result.png``  of three images::

     convert +append image-000.png image-001.png image-003.png result.png

