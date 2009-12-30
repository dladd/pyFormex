.. $Id$  -*- rst -*-
.. pyformex reference manual --- fe
.. CREATED WITH py2rst.py: DO NOT EDIT

.. include:: ../defines.inc
.. include:: ../links.inc

.. _sec:ref-fe:

:mod:`fe` --- Finite Element Models in pyFormex.
================================================

.. automodule:: fe
   :synopsis: Finite Element Models in pyFormex.



   .. autoclass:: Model


      Model objects have the following methods:

      .. automethod:: nnodes()
      .. automethod:: nelems()
      .. automethod:: ngroups()
      .. automethod:: mplex()
      .. automethod:: splitElems(set)
      .. automethod:: elemNrs(group,set)
      .. automethod:: getElems(sets)
      .. automethod:: renumber(old=None,new=None)

**Functions defined in the module fe**

   .. autofunction:: mergeModels(femodels)
   .. autofunction:: checkUniqueNumbers(nrs,nmin=0,nmax=None,error=None)
   .. autofunction:: mergedModel()

   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

