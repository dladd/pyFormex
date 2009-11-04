.. This may look like plain text, but really is -*- rst -*-

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
- New BuMPix image: problems with wide screen monitor
- BuMPix netboot server (can be used to boot in local network): not done yet
- 2D Text rendering: reinstated GLUT since QT is not compatible. GLUT has a very
  limited number of fonts with 1 fixed size. Therefore, more options were added to
  pyFormex. The user can choose the font size and the best approximated size is calculated.
- 3D font manipulations using FTGL: no further work was done. FTGL is not stable yet.
  Since FTGL can be used also in 2D, it could replace GLUT in future.


The good news
-------------
- New documentation project:

  - Using Sphinx and ReST(`ReStructuredText`_)

    - Is also used by Python
    - The package python-sphinx should be installed to produce documentation
    - Text can be entered in very basic text format and the extension can be chosen (.rst)
    - Text can be exported to several formats like .html

  - Reference manual is created automatically from the docstrings in the pyFormex source code

    - Requires *discipline*: docstrings should be legal ReST ! 
    - Need to check current manual and fix all docstrings: 
      only class docstring is included and not init docstring, interpretation of all characters by ReST correct?
    
  - Website and minutes are also created with ReST
    
  - Manual is split up in different documents: refman, tutorial, user-guide, examples, ...

  - Documentation directory: the manual folder will be removed later
    and replaced by the sphinx folder, which is one level up for now
  
  - How to create documentation?

    - Go to sphinx folder
    - ``make html``
    - Documentation is in _build directory

  - Need some decent example layouts

    - Python code: render text like it is and add nice colored box: ``:: text``
    - Render text like it is: ````text````
    - Bullet list: use marker and indentation    
  
  - PROOFREADERS WELCOME !!!!


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
The next meeting will be held at IBiTech in June 2009.


.. The following directive makes sure the targets are included in footnotes.

.. target-notes::

