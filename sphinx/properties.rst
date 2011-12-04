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
  
  

.. include:: defines.inc
.. include:: links.inc

.. _cha:properties:

********************************
Assigning properties to geometry
********************************

*As of version 0.7.1, the way to define properties for elements of the geometry
has changed thoroughly. As a result, the property system has become much more
flexibel and powerful, and can be used for Formex data structures as well as for
TriSurfaces and Finite Element models.*

With properties we mean any data connected with some part of the geometry other
than the coordinates of its points or the structure of points into elements.
Also, values that can be calculated purely from the coordinates of the points
and the structure of the elements are usually not considerer properties.

Properties can e.g. define material characteristics, external loading and
boundary conditions to be used in numerical simulations of the mechanics of a
structure. The properties module includes some specific functions to facilitate
assigning such properties. But the system is general enough to used it for any
properties that you can think of.

Properties are collected in a :class:`PropertyDB` object. Before you can store
anything in this database, you need to create it. Usually, you will start with
an empty database.  ::

   P = PropertyDB()


.. _sec:general-properties:

General properties
------------------

Now you can start entering property records into the database. A property record
is a lot like a Python dict object, and thus it can contain nearly anything. It
is implemented however as a :class:`CascadingDict` object, which means that the
key values are strings and can also be used as attributes to address the value.
Thus, if P is a property record, then a field named key can either be addressed
as P['key'] or as P.key. This implementation was choosen for the convenience of
the user, but has no further advantages over a normal dict object. You should
not use any of the methods of Python's dict class as key in a property record:
it would override this method for the object.

The property record has four more reserved (forbidden) keys: kind, tag, set,
setname and nr. The kind and nr should never be set nor changed by the user.
kind is used internally to distinguish among different kind of property records
(see :ref:`sec:node-properties`). It should only be used to extend the
:class:`PropertyDB` class with new kinds of properties, e.g. in subclasses. nr
will be set automatically to a unique record number. Some application modules
use this number for identification and to create automatic names for property
sets.

The tag, set and setname keys are optional fields and can be set by the user.
They should however only be used for the intended purposes explained hereafter,
because they have a special meaning for the database methods and application
modules.

The tag field can be used to attach an identification string to the property
record. This string can be as complex as the user wants and its interpretation
is completely left to the user. The :class:`PropertyDB` class just provides an
easy way to select the records by their tag name or by a set of tag names. The
set and setname fields are treated further in :ref:`sec:set-and-setname`.

So let's create a property record in our database. The Prop() method does just
that. It also returns the property record, so you can directly use it further in
your code.  ::

   >>> Stick = P.Prop(color='green',name='Stick',weight=25, \
           comment='This could be anything: a gum, a frog, a usb-stick,...'})
   >>> print Stick

     color = green
     comment = This could be anything: a gum, a frog, a usb-stick,...
     nr = 0
     name = Stick
     weight = 25

Notice the auto-generated `nr` field. Here's another example, with a tag::

   >>> author = P.Prop(tag='author',name='Alfred E Neuman',\
           address=CascadingDict({'street':'Krijgslaan', 'city':'Gent','country':'Belgium'}))
   >>> print author

     nr = 1
     tag = author
     name = Alfred E Neuman
     address = 
       city = Gent
       street = Krijgslaan
       country = Belgium

This example shows that record values can be complex structured objects. Notice
how the :class:`CascadingDict` object is by default printed in a very readible
layout, offsetting each lower level dictionary two more postions to the right.

The :class:`CascadingDict` has yet another fine characteristic: if an attribute
is not found in the toplevel, all values that are instances of
:class:`CascadingDict` or :class:`Dict` (but not the normal Python dict) will be
searched for the attribute. If needed, this searching is even repeated in the
values of the next levels, and further on, thus cascading though all levels of
:class:`CascadingDict` structures until the attribute can eventually be found.
The cascading does not proceed through values in a :class:`Dict`. An attribute
that is not found in any of the lower level dictionaries, will return a None
value.

If you set an attribute of a :class:`CascadingDict`, it is always set in the
toplevel. If you want to change lower level attributes, you need to use the full
path to it.  ::

   >>> print author.st
     Krijgslaan
   >>> author.street = 'Voskenslaan'
   >>> print author.street
     Voskenslaan
   >>> print author.address.street
     Krijgslaan
   >>> author.address.street = 'Wiemersdreef'
   >>> print author.address.street
     Wiemersdreef
   >>> author = P.Prop(tag='author',alias='John Doe',\
           address={'city': 'London', 'street': 'Downing Street 10', 'country': 'United Kingdom'})
   >>> print author

     nr = 2
     tag = author
     alias = John Doe
     address = {'city': 'London', 'street': 'Downing Street 10', 'country': 'United Kingdom'} 

In the examples above, we have given a name to the created property records, so
that we could address them in the subsequent print and field assigment
statements. In most cases however, it will be impractical and unnecessary to
give your records a name. They all are recorded in the :class:`PropertyDB`
database, and will exist as long as the database variable lives. There should be
a way though to request selected data from that database. The :meth:`getProp`
method returns a list of records satisfying some conditions. The examples below
show how it can be used. ::

   >>> for p in P.getProp(rec=[0,2]):
           print p.name
   Stick
   John Doe
   >>>  for p in P.getProp(tag=['author']):
           print p.name
   None
   John Doe
   >>>  for p in P.getProp(attr=['name']):
           print p.nr
   0
   2
   >>>  for p in P.getProp(tag=['author'],attr=['name']):
           print p.name
   John Doe

The first call selects records by number: either a single record number or a
list of numbers can be specified. The second method selects records based on the
value of their tag field. Again a single tag value or a list of values can be
specified. Only those records having a 'tag' filed matching any of the values in
the list will be returned. The third selection method is based on the existence
of some attribute names in the record. Here, always a list of attribute names is
required. Records are returned that posess all the attributes in the list,
independent from the value of those attributes. If needed, the user can add a
further filtering based on the attribute values. Finally, as is shown in the
last example, all methods of record selection can be combined. Each extra
condition will narrow the selection further down.


.. _sec:set-and-setname:

Using the  set and  setname fields
----------------------------------

In the examples above, the property records contained general data, not related
to any geometrical object. When working with large geometrical objects (whether
:class:`Formex` or other type), one often needs to specify properties that only
hold for some of the elements of the object.

The set can be used to specify a list of integer numbers identifying a
collection of elements of the geometrical object for which the current property
is valid. Absence of the set usually means that the property is assigned to all
elements; however, the property module itself does not enforce this behavior: it
is up to the application to implement it.

Any record that has a set field, will also have a setname field, whose value is
a string. If the user did not specify one, a set name will be auto-generated by
the system. The setname field can be used in other records to refer to the same
set of elements without having to specify them again. The following examples
will make this clear. ::

   >>> P.Prop(set=[0,1,3],setname='green_elements',color='green')
       P.Prop(setname='green_elements',transparent=True)

   >>> a = P.Prop(set=[0,2,4,6],thickness=3.2)
       P.Prop(setname=a.setname,material='steel')

   >>> for p in P.getProp(attr=['setname']):
           print p

   color = green
   nr = 3
   set = [0 1 3]
   setname = green_elements

   nr = 4
   transparent = True
   setname = green_elements

   nr = 5
   set = [0 2 4 6]
   setname = Set_5
   thickness = 3.2

   nr = 6
   material = steel
   setname = Set_5

In the first case, the user specifies a setname himself. In the second case, the
auto-generated name is used. As a convenience, the user is allowed to write
set=name instead of setname=name when referring to an already defined set.   ::

   >>> P.Prop(set='green_elements',transparent=False)
       for p in P.getProp(attr=['setname']):
           if p.setname == 'green_elements':
               print p.nr,p.transparent

   3 None
   4 True
   7 False

Record 3 does not have the transparent attribute, so a value None is printed.


.. _sec:special-properties:

Specialized property records
----------------------------

The property system presented above allows for recording any kind of values. In
many situations however we will want to work with a specialised and limited set
of attributes. The main developers of e.g. often use the program to create
geometrical models of structures of which they want to analyse the mechanical
behavior. These numerical simulations (FEA, CFD) require specific data that
support the introduction of specialised property records. Currently there are
two such property record types: node properties (see
:ref:`sec:node-properties`), which are attributed to a single point in space,
and element properties (:ref:`sec:elem-properties`), which are attributed to a
structured collection of points.

Special purpose properties are distincted by their kind field. General property
records have kind=", node properties haven kind='n' and  kind='e' is set for
element properties. Users can create their own specialised property records by
using other value for the kind parameter.


.. _sec:node-properties:

Node properties
---------------

Node properties are created with the :meth:`nodeProp` method, rather than the
general :meth:`Prop`. The kind field does not need to be set: it will be done
automatically. When selecting records using the :meth:`getProp` method, add a
kind='n' argument to select only node properties.

Node properties will recognize some special field names and check the values for
consistency. Application plugins such as the Abaqus input file generator depend
on these property structure, so the user should not mess with them. Currently,
the following attributes are in use:

cload
   A concentrated load at the node. This is a list of 6 items: three force
   components in axis directions and three force moments around the axes: [F_0,
   F_1, F_2, M_0, M_1, M_2].

bound
   A boundary condition for the nodal displacement components. This can be defined
   in 2 ways:

* as a list of 6 items [ u_0, u_1, u_2, r_0, r_1, r_2 ]. These items have 2
     possible values:

     0
        The degree of freedom is not restrained.

     1
        The degree of freedom is restrained.

* as a string. This string is a standard boundary type. Abaqus will recognize
     the following strings:

* PINNED

* ENCASTRE

* XSYMM

* YSYMM

* ZSYMM

* XASYMM

* YASYMM

* ZASYMM

displacement
   Prescribed displacements. This is a list of tuples (i,v), where i is a DOF
   number (1..6) and v is the prescribed value for that DOF.

coords
   The coordinate system which is used for the definition of cload, bound and displ
   fields. It should be a :class:`CoordSys` object.

Some simple examples::

   P.nodeProp(cload=[5,0,-75,0,0,0])
   P.nodeProp(set=[2,3],bound='pinned')
   P.nodeProp(5,displ=[(1,0.7)])

The first line sets a concentrated load all the nodes, the second line sets a
boundary condition 'pinned' on nodes 2 and 3. The third line sets a prescribed
displacement on node 5 with value 0.7 along the first direction. The first
positional argument indeed corresponds to the 'set' attribute.

Often the properties are computed and stored in variables rather than entered
directly.  ::

   P1 = [ 1.0,1.0,1.0, 0.0,0.0,0.0 ]
   P2 = [ 0.0 ] * 3 + [ 1.0 ] * 3 
   B1 = [ 1 ] + [ 0 ] * 5
   CYL = CoordSystem('cylindrical',[0,0,0,0,0,1])
   P.nodeProp(bound=B1,csys=CYL)

The first two lines define two concentrated loads: ``P1`` consists of three
point loads in each of the coordinate directions; P2 contains three force
moments around the axes. The third line specifies a boundary condition where the
first DOF (usually displacement in :math:`x`\ -direction) is constrained, while
the remaining 5 DOF's are free. The next line defines a local coordinate system,
in this case a cylindrical coordinate system with axis pointing from point
``[0.,0.,0.]`` to point ``[0.,0.,1.]``. The last line

To facilitate property selection, a tag can be added.  ::

   nset1 = P.nodeProp(tag='loadcase 1',set=[2,3,4],cload=P1).nr
   P.nodeProp(tag='loadcase 2',set=Nset(nset1),cload=P2)

The last two lines show how you can avoid duplication of sets in mulitple
records. The same set of nodes should receive different concentrated load values
for different load cases. The load case is stored in a tag, but duplicating the
set definition could become wasteful if the sets are large. Instead of
specifying the node numbers of the set directly, we can pass a string setting a
set name. Of course, the application will need to know how to interprete the set
names. Therefore the property module provides a unified way to attach a unique
set name to each set defined in a property record. The name of a node property
record set can be obtained with the function Nset(nr), where nr is the record
number. In the example above, that value is first recorded in nset1 and then
used in the last line to guarantee the use of the same set as in the property
above.


.. _sec:elem-properties:

Element properties
------------------

The :meth:`elemProp` method creates element properties, which will have their
``kind`` attribute set to 'e'. When selecting records using the :meth:`getProp`
method, add the kind='e' argument to get element properties.

Like node properties, element property records have a number of specialize
fields. Currently, the following ones are recognized by the Abaqus input file
generator.

eltype
   This is the single most important element property. It sets the element type that
   will be used during the analysis. Notice that a Formex object also may have an
   ``eltype`` attribute; that one however is only used to describe the type of the
   geometric elements involved. The element type discussed here however may also
   define some other characteristics of the element, like the number and type of
   degrees of freedom to be used in the analysis or the integration rules to be
   used. What element types are available is dependent on the analysis package to
   be used. Currently, does not do any checks on the element type, so the
   simulation program's own element designation may be used.

section
   The section properties of the element. This should be an :class:`ElemSection`
   instance, grouping material properties (like Young's modulus) and geometrical
   properties (like plate thickness or beam section).

dload
   A distributed load acting on the element. The value is an :class:`ElemLoad`
   instance. Currently, this can include a label specifying the type of distributed
   loading, a value for the loading, and an optional amplitude curve for specifying
   the variation of a time dependent loading.


Property data classes
---------------------

The data collected in property records can be very diverse. At times it can
become quite difficult to keep these data consistent and compatible with other
modules for further processing. The property module contains some data classes
to help you in constructing appropriate data records for Finite Element models.
The FeAbq module can currently interprete the following data types.

:class:`CoordSystem` defines a local coordinate system for a node. Its
constructor takes two arguments:

* a string defining the type of coordinate system, either 'Rectangular',
  'Cylindrical' or 'Spherical' (the first character suffices), and

* a list of 6 coordinates, specifying two points A and B. With 'R', A is on the
  new :math:`x`\ -axis and B is on the new ':math:`y` axis. With 'C' and 'S', AB
  is the axis of the cylindrical/spherical coordinates.

Thus, ``CoordSystem('C',[0.,0.,0.,0.,0.,1.])`` defines a cylindrical coordinate
system with the global :math:`z` as axis.

:class:`ElemLoad` is a distributed load on an element. Its constructor takes two
arguments:

* a label defining the type of loading,

* a value for the loading,

* optionally, the name of an amplitude curve.

E.g., ElemLoad('PZ',2.5) defines a distributed load of value 2.5 in the
direction of the :math:`z`\ -axis.

:class:`ElemSection` can be used to set the material and section properties on
the elements. It can hold:

* a section,

* a material,

* an optional orientation,

* an optional connector behavior,

* a sectiontype (deprecated). The sectiontype should preferably be set togehter
  with the other section parameters.

An example::

   >>> steel = {
       'name': 'steel',
       'young_modulus': 207000,
       'poisson_ratio': 0.3,
       'density': 0.1,
       }
   >>> thin_plate = { 
       'name': 'thin_plate',
       'sectiontype': 'solid',
       'thickness': 0.01,
       'material': 'steel',
       }
   >>> P.elemProp(eltype='CPS3',section=ElemSection(section=thin_plate,material=steel))

First, a material is defined. Then a thin plate section is created, referring to
that material. The last line creates a property record that will attribute this
element section and an element type 'CPS3' to all elements.


Exporting to finite element programs
====================================

.. End
