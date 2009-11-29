.. $Id$  -*- rst -*-
.. pyformex reference manual --- lima
.. CREATED WITH py2rst.py: DO NOT EDIT

.. include:: ../defines.inc
.. include:: ../links.inc

.. _sec:ref-lima:

:mod:`lima` --- Lindenmayer Systems
===================================

.. automodule:: lima
   :synopsis: Lindenmayer Systems



   .. autoclass:: Lima


      Lima objects have the following methods:

      .. automethod:: status()
      .. automethod:: addRule(atom,product)
      .. automethod:: translate(rule,keep=False)
      .. automethod:: grow(ngen=1)

**Functions defined in the module lima**

   .. autofunction:: lima(axiom,rules,level,turtlecmds,glob=None)

   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

