.. $Id$

..
  This file is part of pyFormex 0.8.9  (Fri Nov  9 10:49:51 CET 2012)
  pyFormex is a tool for generating, manipulating and transforming 3D
  geometrical models by sequences of mathematical operations.
  Home page: http://pyformex.org
  Project page:  http://savannah.nongnu.org/projects/pyformex/
  Copyright 2004-2012 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be)
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

pyFormex Geometry File Format 1.6
=================================
This describes the pyFormex Geometry File Format (PGF) version 1.6 as
drafted on 2013-03-10 and being used in pyFormex 0.9.0.
The version numbering is such that implementations of a later
version are able to read an older version with the same major numbering.
Thus, the 1.6 version can still read version 1.5 files.

The prefered filename extension for pyFormex geometry files is '.pgf',
though this is not a requirement.

General principles
------------------

The PGF format consists of a sequence of records of two types: comment
lines and data blocks. A record always ends with a newline character,
but not all newline characters are record separators: data blocks may
include multiple newlines as part of the data.

Comment records are ascii and start with a '#' character. Comment records
are mostly used to announce the type and amount of data in the following
data block(s). This is done by comment line containing a sequence of
'key=value' statements, separated by semicolons (';').

Data blocks can be either ascii or binary, and are always announced by
specially crafted comment lines preceding them. Note that even binary
data blocks get a newline character at the end, to mark the end of the
record.


Detailed layout
---------------

The pyFormex Geometry File starts with a header comment line identify
the file type and version, and possibly specifying some global variables.
For the version 1.6 format the first line may look like::

  # pyFormex Geometry File (http://pyformex.org) version='1.6'; sep=' '

The version number is used to read back legacy formats in newer versions
of pyFormex. The `sep = ' '` defines the default data separator for
data blocks that do not specify it (see below).


The remainder of the file is a sequence of comment lines announcing
data blocks, followed by those data blocks. The announcement line
provides information about the number, type and size of data blocks
that follow. This makes it possible to write and read the data using
high speed functions (like `numpy.tofile` and `numpy.fromfile`) and without
having to test any contents of the data.
The data block information in the announcement line is provided by a number
of 'key=value' strings separated with a semicolon and optional whitespace.


Object type specific fields
...........................
For each object type that can be stored, there are some required fields
and data blocks. In the examples below, `<int>` stands for an integer number,
`<str>` for a string, and `<bool>` for either `True` or `False`.

- Formex: the announcement provides at least::

    # objtype='Formex'; nelems=<int>; nplex=<int>

  The data block following this line should contain exactly `nelems*nplex*3`
  floating point values: the 3 coordinates of the `nplex` points of the
  `nelems` elements of the Formex.

- Mesh: the announcement contains at least::

    # objtype='Mesh'; ncoords=<int>; nelems=<int>; nplex=<int>

  In this case two data blocks will follow: first `ncoords*3` float values
  with the coordinates of the nodes; then a block with `nelems*nplex`
  integer values: the connectivity table of the mesh.

- Curve:

Optional fields
...............
The announcement line may contain other fields, usually to define extra
attributes for the object:

- `props=<bool>` : If the value is True, another data block with `nelems`
  integer values follows. These are the property numbers of the object.

- `eltype=<str>` : Can also have the special value None. If specified and
  not None, it will be used to set the element type of the object.

- `name=<str>` : Name of the object. If specified, pyFormex will use this
  value as a key when returning the restored object.

- `sep=<str>` : This field defines how the data are stored. If it is not
  defined, the value from the file header is used.

  - An empty string means that the data blocks are written in binary.
    Floating point values are stored as little-endian 4byte floats, while
    integer values are stored as 4 byte integers.

  - Any other string makes the data being written in ascii mode, with the
    specified string used as a separator between any two values. When
    reading a PGF file, extra whitespace and newlines appearing around the
    separator are silently ignored.



Example
-------

The following pyFormex script creates a PGF file containing two objects,
a Formex with one square, and a Mesh with two triangles::

  F = Formex('4:0123')
  M = Formex('3:112.34').setProp(1).toMesh()
  writeGeomFile('test.pgf',[F,M],sep=', ')

The Mesh has property numbers defined on it, the Formex doesn't.
The data are written in ascii mode with ', ' as separator.
Here is the resulting contents of the file 'test.pgf'::

  # pyFormex Geometry File (http://pyformex.org) version='1.6'; sep=', '
  # objtype='Formex'; nelems=1; nplex=4; props=False; eltype=None; sep=', '
  0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0
  # objtype='Mesh'; ncoords=4; nelems=2; nplex=3; props=True; eltype='tri3'; sep=', '
  1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 0.0
  0, 1, 3, 3, 2, 0
  1, 1

This file contains two objects: a Formex and a Mesh. The Formex has 1 element
of plexitude 4 and no property numbers. Following its announcement is a single
data block with 1x4x3 = 12 coordinate values.
The Mesh contains 2 elements of plexitude 3, has element type 'tri3' and
contains property numbers. Following the announcement are three data blocks:
first the 4*3 nodal coordinates, then the 2*3 = 6 entries in the connectivity
table, and finally 2 property numbers.

.. End
