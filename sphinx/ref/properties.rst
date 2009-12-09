.. $Id$  -*- rst -*-
.. pyformex reference manual --- properties
.. CREATED WITH py2rst.py: DO NOT EDIT

.. include:: ../defines.inc
.. include:: ../links.inc

.. _sec:ref-properties:

:mod:`properties` --- General framework for attributing properties to geometrical elements.
===========================================================================================

.. automodule:: properties
   :synopsis: General framework for attributing properties to geometrical elements.



   .. autoclass:: Database


      Database objects have the following methods:

      .. automethod:: readDatabase(filename,None)

   .. autoclass:: MaterialDB


      MaterialDB objects have the following methods:


   .. autoclass:: SectionDB


      SectionDB objects have the following methods:


   .. autoclass:: ElemSection


      ElemSection objects have the following methods:

      .. automethod:: addSection(section)
      .. automethod:: computeSection(section)
      .. automethod:: addMaterial(material)

   .. autoclass:: ElemLoad


      ElemLoad objects have the following methods:


   .. autoclass:: EdgeLoad


      EdgeLoad objects have the following methods:


   .. autoclass:: CoordSystem


      CoordSystem objects have the following methods:


   .. autoclass:: Amplitude


      Amplitude objects have the following methods:


   .. autoclass:: PropertyDB


      PropertyDB objects have the following methods:

      .. automethod:: autoName(clas,kind)
      .. automethod:: setMaterialDB(aDict)
      .. automethod:: setSectionDB(aDict)
      .. automethod:: Prop(kind='',tag=None,set=None,name=None)
      .. automethod:: getProp(kind='',rec=None,tag=None,attr=[],noattr=[],delete=False)
      .. automethod:: delProp(kind='',rec=None,tag=None,attr=[])
      .. automethod:: nodeProp(prop=None,set=None,name=None,tag=None,cload=None,bound=None,displ=None,csys=None,ampl=None)
      .. automethod:: elemProp(prop=None,grp=None,set=None,name=None,tag=None,section=None,eltype=None,dload=None,eload=None,ampl=None)

**Functions defined in the module properties**

   .. autofunction:: checkIdValue(values)
   .. autofunction:: checkArrayOrIdValue(values)
   .. autofunction:: checkString(a,valid)
   .. autofunction:: autoName(base)
   .. autofunction:: Nset()
   .. autofunction:: Eset()
   .. autofunction:: FindListItem(l,p)
   .. autofunction:: RemoveListItem(l,p)

   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

