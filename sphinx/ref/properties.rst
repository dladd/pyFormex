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
   :members: checkIdValue,checkArrayOrIdValue,checkString,FindListItem,RemoveListItem

   ``Classes defined in module properties``


   .. autoclass:: Amplitude
      :members: 

   .. autoclass:: CoordSystem
      :members: 

   .. autoclass:: Database
      :members: readDatabase,update,get,setdefault

   .. autoclass:: EdgeLoad
      :members: update,get,setdefault

   .. autoclass:: ElemLoad
      :members: update,get,setdefault

   .. autoclass:: ElemSection
      :members: addSection,computeSection,update,get,addMaterial,setdefault

   .. autoclass:: MaterialDB
      :members: readDatabase,update,get,setdefault

   .. autoclass:: PropertyDB
      :members: update,get,setdefault,setMaterialDB,setSectionDB,Prop,getProp,delProp,nodeProp,elemProp

   .. autoclass:: SectionDB
      :members: readDatabase,update,get,setdefault

   ``Functions defined in module properties`` 


   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

