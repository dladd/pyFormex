.. This may look like plain text, but really is -*- rst -*-
  
..
  This file is part of the pyFormex project.
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
  
  

.. include:: links.inc

====================================
pyFormex User Meeting 9 (2009-11-03)
====================================


Place and Date
==============
These are the minutes of the pyFormex User Meeting of Thursday January 29, 2009, at the Institute Biomedical Technology (IBiTech), Ghent University, Belgium.


Participants
============
The following pyFormex developers, users and enthusiasts were present.

- Benedict Verhegghe
- Matthieu De Beule
- Peter Mortier
- Sofie Van Cauter, secretary
- Gianluca De Santis
- Tomas Praet


Apologies
=========
- None


Minutes of the previous meeting
===============================
- The minutes of the previous meeting were approved and will be put on the `pyFormex User Meeting page`_.


Agenda and discussion
=====================

The bad news
------------
- Then new BuMPix image (0.5) has problems with newer graphics cards
  and wide screen monitors. *Update: a newer image 0.5.1 has been
  created that resolves those problems.*
- The promised BuMPix netboot server (which can be used to boot BuMPix
  in local network) has not been done yet.
- 2D Text rendering: Because the QT4 funcdtionality for drawing 2D
  text on an OpenGL canavas is not compatible with many parts of the
  pyFormex endering machine, we had to reinstate GLUT for 2D text
  drawing. Sadly, this means a very limited number of fonts and
  fontsizes. Some functions were added to allow easily picking the
  GLUT font/size that comes close to the expectations.
- 3D font manipulations using FTGL: no further work was done.
  When fully developed, this could probably be used for 2D text drawing
  as well, instead of the limited  GLUT fonts.


The good news
-------------

- The properties database and export to Abaqus
  
  - setname -> name

    ::

      PDB.elemProp(set=[1,2,5,6],name='set1',....)
      PDB.elemProp(name='set1',....)
      PDB.elemProp(set='set1',....)
      
    - ``'set1'`` is defined in line 1 and is used in line 2 and 3
    - Line 2 and 3 are interpreted the same
  
  - any new fields allowed also in elemProp, nodeProp via ``**kargs``

    ::

      PDB.elemProp(...,myproperty='value')

      
- Revised Input Dialogs 

  - All items can display a string other than the name of the
    input/return value.
  
  - Old item format 

    ::
      
      ( name, value, [type, [other args, [...., [options ]]]] )

  - New item format 

    ::

      ( name, value, [type], [options] )      
      
    where ``options`` is a dictionary
    
    :text: the text to be displayed (ALL)
    :choices: list of items from which one can be selected (select,radio)
    :min, max: limits (int, float)
    :buttons: a list of (buttonname,function) tuples
    
    The old format will still work for now
    
  - Tabbed Input Dialogs: items is a dictionary 

    ::

      { 'page one': itemlist, 'page two': itemlist }

    where ``itemlist`` is the same as ``items`` for a single page

  - utils.subDict(dict,keystart): returns all items in dict whose key
    starts with keystart. Use e.g. in tabbed dialogs to get results
    from a specific page.

  - Items can have (a) button(s) (e.g. to set the value)

- New documentation project:

  - Using Sphinx and ReST(`ReStructuredText`_)

    - Text can be entered in very basic text format and the extension can be chosen (.rst)
    - Text can be exported to several formats like .html 
    - Is also used by Python documentation
    - pyFormex website and minutes are also created with ReST

  - Reference manual is created semi-automatically from the docstrings
    in the pyFormex source code

    - Requires *discipline*: docstrings should be legal ReST ! 
    - Need to check current manual and fix all docstrings: 
      only class docstring is included and not init docstring, interpretation of all characters by ReST correct?
    
  - Manual is split up in different documents: refman, tutorial, user-guide, examples, ...

  - Documentation directory: the manual folder will be removed later
    and replaced by the sphinx folder, which is one level up for now
  
  - How to create documentation?

    - The package `python-sphinx` should be installed
    - Go to sphinx folder
    - ``make html``
    - Documentation is in _build directory

  - We need some decent example layouts for common cases

    - Bullet list: use marker and indentation    
    - Render text like it is: ````text````
    - Python code: render text like it is and add nice colored box: ``:: text``
  
  - All pyFormex users are invited to read the documentation and make
    comments or contribute.

- ColorImage example: creates a color and colormap from an image file
  and performes some transformations on it

- TODO for version 1.0 (aimed end 2010): users can make suggestions

Varia
=====

- FEops realizes first commercial project with pyFormex
- Is it possible to run pyFormex via Athena? It should be checked which software
  is needed for Windows or we should use a browser and run pyFormex on a server
- Parallel computing will be implemented after version 1.0
- Developers are encouraged to add new interesting general functions before version 1.0


Date of the next meeting
========================
The next meeting will be held at IBiTech in Jan 2010.


.. The following directive makes sure the targets are included in footnotes.

.. target-notes::

