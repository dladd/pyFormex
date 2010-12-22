.. $Id$  -*- rst -*-
.. pyformex reference manual --- curve
.. CREATED WITH py2rst.py: DO NOT EDIT

.. include:: <isonum.txt>
.. include:: ../defines.inc
.. include:: ../links.inc

.. _sec:ref-curve:

:mod:`curve` --- Definition of curves in pyFormex.
==================================================

.. automodule:: curve
   :synopsis: Definition of curves in pyFormex.
   :members: circle,convertFormexToCurve

   ``Classes defined in module curve``


   .. autoclass:: Curve
      :members: pointsOn,pointsOff,ncoords,npoints,sub_points,sub_points_2,sub_directions,sub_directions_2,lengths,pointsAt,directionsAt,subPoints,length,approx,toFormex

   .. autoclass:: PolyLine
      :members: nelems,toFormex,toMesh,sub_points,sub_points2,sub_directions,vectors,directions,avgDirections,lengths,atLength,reverse,split,cutWithPlane,distanceOfPoints,distanceOfPolyLine

   .. autoclass:: Polygon
      :members: 

   .. autoclass:: BezierSpline
      :members: pointsOn,pointsOff,part,sub_points,sub_directions,length_intgrnd,lengths,approx_by_subdivision,extend,reverse

   .. autoclass:: CardinalSpline
      :members: 

   .. autoclass:: CardinalSpline2
      :members: sub_points

   .. autoclass:: NaturalSpline
      :members: compute_coefficients,sub_points

   .. autoclass:: Arc3
      :members: sub_points

   .. autoclass:: Arc
      :members: sub_points

   .. autoclass:: Spiral
      :members: 

   .. autoclass:: QuadBezierSpline
      :members: 

   ``Functions defined in module curve`` 


   
.. moduleauthor:: pyFormex project (http://pyformex.org)

.. End

