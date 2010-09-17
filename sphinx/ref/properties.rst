.. $Id$  -*- rst -*-
.. pyformex reference manual --- properties
.. CREATED WITH py2rst.py: DO NOT EDIT

.. include:: <isonum.txt>
.. include:: ../defines.inc
.. include:: ../links.inc

.. _sec:ref-properties:

:mod:`properties` --- General framework for attributing properties to geometrical elements.
===========================================================================================

.. automodule:: properties
   :synopsis: General framework for attributing properties to geometrical elements.



   .. autoclass:: Database
      :members: readDatabase

   .. autoclass:: MaterialDB
      :members: 

   .. autoclass:: SectionDB
      :members: 

   .. autoclass:: ElemSection
      :members: addSection,computeSection,addMaterial

   .. autoclass:: ElemLoad
      :members: 

   .. autoclass:: EdgeLoad
      :members: 

   .. autoclass:: CoordSystem
      :members: 

   .. autoclass:: Amplitude
      :members: 

   .. autoclass:: PropertyDB
      :members: setMaterialDB,setSectionDB,Prop,getProp,delProp,nodeProp,elemProp

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

