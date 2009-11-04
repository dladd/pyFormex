.. This may look like plain text, but really is -*- rst -*-

.. include:: links.inc

====================================
pyFormex User Meeting 8 (2009-11-03)
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
- Sofie Van Cauter
- Gianluca De Santis
- Tomas Praet


Apologies
=========
- None


Minutes of the previous meeting
===============================
- The minutes of the previous meeting were not yet available. They will be approved later and put on the `pyFormex User Meeting page`_.


Agenda and discussion
=====================

The bad news
------------
- new BuMPix image (problems with wide screen)
- BuMPix netboot server (not)
- 2D Text rendering (reinstated GLUT, QT not compatible)
- 3D font manipulations using FTGL (no further work was done)

The good news
-------------
- New documentation project:

  - using Sphinx and ReST(`ReStructuredText`_): full
    reference manual can be created automatically.
    Requires *discipline*: docstrings should be legal ReST !

  - need to check current manual and fix all docstrings

  - need some decent example layouts

  - PROOFREADERS WELCOME !!!!
 
- The properties database and export to Abaqus

  - setname -> name

    ::

      PDB.EProp(set=[1,2,5,6],name='set1',....)
      PDB.EProp(name='set1',....)
      PDB.EProp(set='set1',....)
  
  - any new fields allowed also in EProp, NProp

    ::

      PDB.EProp(...,myproperty='value')


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
    
  - Tabbed Input Dialogs: items is a dictionary 

    ::

      { 'page one': itemlist, 'page two': itemlist }

  - utils.subDict(dict,keystart): returns all items in dict whose key
    starts with keystart. Use e.g. in tabbed dialogs to get results
    from a specific page.

  - Items can have (a) button(s) (e.g. to set the value)

- ColorImage example

- TODO for version 1.0 (aimed end 2010)

Varia
=====

- FEops realizes first commercial project with pyFormex


Date of the next meeting
========================
The next meeting will be held at IBiTech in June 2009.


.. The following directive makes sure the targets are included in footnotes.

.. target-notes::

