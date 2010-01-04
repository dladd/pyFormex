.. $Id$
.. pyformex manual --- file format

.. include:: defines.inc
.. include:: links.inc


.. _cha:fileformat:
.. sectionauthor:: Benedict Verhegghe <benedict.verhegghe@ugent.be>

*********************
pyFormex file formats
*********************

:Date: |today|
:Version: |version|
:Author: Benedict Verhegghe <benedict.verhegghe@ugent.be>

.. topic:: Abstract

  This document describes the native file formats used by pyFormex.
  The are currently two file formats: the pyFormex Project File (.pyf)
  and the pyFormex Geometry File (.pgf/.formex).


Introduction
============
pyFormex uses two native file formats to save data on a persistent medium: 
the pyFormex Project File (.pyf) and the pyFormex Geometry File (.pgf).

A Project File can store any pyFormex data and is the prefered way 
to store your data for later reuse within pyFormex. The data in the resulting
file can normally not be used by humans and can only be easily restored
by pyFormex itself.

The pyFormex Geometry File on the other hand can be used to exchange
data between pyFormex projects or with other software. Because of its
plain text format, the data can be read and evend edited by humans.
You may also wish to save data in this format to make them accessible
the need for pyFormex, or to bridge incompatible changes in pyFormex.

Because the geometrical data in pyFormex can be quite voluminous, the 
format has been chosen so as to allow efficient read and write operations
from inside pyFormex. If you want a nicer layout and efficiency is not your
concern, you can used the :meth:`fprint` method of the geometry object.


.. _sec:project_file_format:

pyFormex Project File Format
============================

A pyFormex project file is just a pickled Python dictionary stored on file,
possibly with compression. Any pyFormex objects can be exported and stored on
the project file. The resulting file is normally not readable for humans and
because all the class definitions of the exported data have to be present,
the file can only be read back by pyFormex itself. 

The format of the project file is therefore currently not further documented.
See :doc:`projects` for the use of project files from within pyFormex.




.. _sec:geometry_file_format:

pyFormex Geometry File Format 1.2
=================================
This describes the pyFormex Geometry File Format version 1.2 as drawn on
2010-01-04. The version numbering is such that implementations of a later
version are able to read an older version with the same major numbering.
Thus, the 1.2 version still can read version 1.0 and 1.1 files.

The prefered filename extension for pyFormex geometry files is '.pgf', 
though this is not a requirement and the previously used '.formex' is
certainly as valid as any other.

The pyFormex Geometry File starts with a header line identify the file
type and version, and possibly specifying some global variables::
  
  # Formex File Format 1.1 (http://pyformex.org)
  
The remainder of the file contains one or more data blocks, each of which 
consists of a header line followed by the numerical data. The header line
starts with a '#'. The remainder of the line is a sequence of 'keyword=value'
strings separated with a semicolon and optional whitespace, such as in the 
following example::

  # nelems=692; nplex=2; props=True; eltype=None; sep=' '
 
The keywords in the data header specify the type and amount of that that
will be read, and how they will be structured in arrays and converted to
pyFormex objects. The example above specifies a Formex of plexitude 2 having
692 elements with no specific element type but possessin property numbers.
The separator used in the data is a single space.

- comment lines, stating with a '#', but maybe holding invaluable information
  for the interpretation of the rest of the data,
- data blocks.

Data blocks are written using :func:`numpy.tofile` and read back with
:func:`numpy.fromfile`. All data items in a block are of the same type
and are written as ASCII strings, separated by a constant string. The
separator can be specified by the user and defaults to a single space,
so that all data of a single block are written on one line, separated
by a blank. When reading, newline characters will be silently ignored
or used as a separator character as well. As a special case, if an
empty string is specified as separator, the data will be written in
binary mode.

Each data block is preceded with a comments line with the following structure:



.. End
