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
   :members: checkIdValue,checkArrayOrIdValue,checkString,autoName,Nset,Eset,FindListItem,RemoveListItem

   ``Classes defined in module properties``


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

   ``Functions defined in module properties`` 


   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

