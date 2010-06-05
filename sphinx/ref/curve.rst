.. $Id$  -*- rst -*-
.. pyformex reference manual --- curve
.. CREATED WITH py2rst.py: DO NOT EDIT

.. include:: ../defines.inc
.. include:: ../links.inc

.. _sec:ref-curve:

:mod:`curve` --- Definition of curves in pyFormex.
==================================================

.. automodule:: curve
   :synopsis: Definition of curves in pyFormex.



   .. autoclass:: Curve


      Curve objects have the following methods:

      .. automethod:: pointsOn()
      .. automethod:: pointsOff()
      .. automethod:: ncoords()
      .. automethod:: npoints()
      .. automethod:: sub_points(t,j)
      .. automethod:: sub_points_2(t,j)
      .. automethod:: lengths()
      .. automethod:: pointsAt(t)
      .. automethod:: subPoints(div=10,extend=[0.,0.])
      .. automethod:: length()
      .. automethod:: approx(ndiv=None,ntot=None)
      .. automethod:: toFormex(None)

   .. autoclass:: PolyLine


      PolyLine objects have the following methods:

      .. automethod:: nelems()
      .. automethod:: toFormex()
      .. automethod:: toMesh()
      .. automethod:: sub_points(t,j)
      .. automethod:: sub_points2(t,j)
      .. automethod:: vectors()
      .. automethod:: directions(return_doubles=False)
      .. automethod:: avgDirections(return_doubles=False)
      .. automethod:: lengths()
      .. automethod:: atLength(div)
      .. automethod:: reverse()
      .. automethod:: split(i)
      .. automethod:: cutWithPlane(p,n,side='')

   .. autoclass:: Polygon


      Polygon objects have the following methods:


   .. autoclass:: BezierSpline


      BezierSpline objects have the following methods:

      .. automethod:: pointsOn()
      .. automethod:: pointsOff()
      .. automethod:: sub_points(t,j)

   .. autoclass:: QuadBezierSpline


      QuadBezierSpline objects have the following methods:

      .. automethod:: pointsOn()
      .. automethod:: pointsOff()
      .. automethod:: sub_points(t,j)

   .. autoclass:: CardinalSpline


      CardinalSpline objects have the following methods:


   .. autoclass:: CardinalSpline2


      CardinalSpline2 objects have the following methods:

      .. automethod:: compute_coefficients()
      .. automethod:: sub_points(t,j)

   .. autoclass:: NaturalSpline


      NaturalSpline objects have the following methods:

      .. automethod:: compute_coefficients()
      .. automethod:: sub_points(t,j)

   .. autoclass:: Arc3


      Arc3 objects have the following methods:

      .. automethod:: sub_points(t,j)

   .. autoclass:: Arc


      Arc objects have the following methods:

      .. automethod:: sub_points(t,j)

   .. autoclass:: Spiral


      Spiral objects have the following methods:


**Functions defined in the module curve**

   .. autofunction:: vectorPairAngle(v1,v2)
   .. autofunction:: convertFormexToCurve(closed=False)

   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

